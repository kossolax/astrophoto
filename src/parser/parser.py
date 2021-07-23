import cv2
import rawpy
import numpy as np
import struct

from abc import ABC, abstractmethod

class Parser(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def read(self, path: str) -> np.array:
        pass

class ParserCV2(Parser):
    def __init__(self):
        pass

    def read(self, path: str) -> np.array:
        return cv2.imread(path)

class ParserRAW(Parser):
    def __init__(self):
        pass

    def read(self, path: str) -> np.array:
        with rawpy.imread(path) as raw:
            #rgb = raw.postprocess(gamma=(2.222, 4.5), no_auto_bright=True, output_bps=8, half_size=True)
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, half_size=True)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

class ParserTIF(Parser, ABC):
    def __init__(self):
        super().__init__()
        self.TAGName = {
            0x00FE: "NewSubFileType",
            0x0100: "ImageWidth",
            0x0101: "ImageLength",
            0x0102: "BitsPerSample",
            0x0103: "Compression",

            0x010F: "Make",
            0x0110: "Model",

            0x0111: "StripOffsets",
            0x0112: "Orientation",
            0x0115: "SamplesPerPixel",
            0x0116: "RowsPerStrip",
            0x0117: "StripByteCounts",

            0x011A: "XResolution",
            0x011B: "YResolution",
            0x011C: "PlanarConfiguration",
            0x0128: "ResolutionUnit",

            0x0132: "DateTime",
            0x013B: "Artist",
            0x014a: "SubIFDs",

            0x829a: "ExposureTime",
            0x829d: "FNumber",
            0x8822: "ExposureProgram",
            0x8827: "ISOSpeedRatings",
            0x8830: "SensitivityType",
            0x828E: "CFAPattern",

            0x9203: "BrightnessValue",
            0x9204: "ExposureBiasValue",
            0x9205: "MaxApertureValue",
            0x920a: "FocalLength",

        }

    def TIF_HEAD(self, ptr):
        id, ver, off = struct.unpack("HHL", ptr.read(2 + 2 + 4))
        if id != 0x4949 and ver != 0x2a:
            raise Exception("unsupported file format")
        return id, ver, off

    def TIF_IFD(self, ptr, goto):
        ptr.seek(goto)
        num = struct.unpack("H", ptr.read(2))[0]
        tags = []
        for i in range(0, num):
            tags.append(self.TIF_TAG(ptr))
        offset = struct.unpack("L", ptr.read(4))[0]
        return tags, offset

    def TIF_TAG(self, ptr):
        id, type = struct.unpack("HH", ptr.read(2 + 2))
        count = struct.unpack("L", ptr.read(4))[0]
        offset = struct.unpack("L", ptr.read(4))[0]
        return id, type, count, offset

    def TIF_TAG_PARSE(self, ptr, tag):
        ret = {}

        for id, type, count, offset in tag:
            value = offset
            ptr.seek(offset)

            if type == 1:  # BYTE
                if count > 1:
                    value = []
                    for i in range(0, count):
                        value.append(struct.unpack("B", ptr.read(1))[0])
            elif type == 2:  # ASCII
                value = ""
                b = struct.unpack("B", ptr.read(1))[0]
                while b != 0:
                    value += chr(b)
                    b = struct.unpack("B", ptr.read(1))[0]
            elif type == 3:  # SHORT
                if count > 1:
                    value = []
                    for i in range(0, count):
                        value.append(struct.unpack("H", ptr.read(2))[0])
            elif type == 4:  # LONG
                if count > 1:
                    value = []
                    for i in range(0, count):
                        value.append(struct.unpack("L", ptr.read(4))[0])
            elif type == 5:  # RATIONAL
                value = struct.unpack("LL", ptr.read(4 + 4))
                pass
            elif type == 6:  # SBYTE
                value = struct.unpack("b", ptr.read(1))[0]
                pass
            elif type == 7:  # UNDEFINED
                value = struct.unpack("c", ptr.read(1))[0]
                pass
            else:
                # raise Exception("Unsupported data type: " + str(type))
                pass

            if id in self.TAGName:
                ret[self.TAGName[id]] = value
            else:
                pass

        return ret

    @abstractmethod
    def TIF_STRIP(self, ptr, data):
        pass

    @abstractmethod
    def read(self, path: str) -> np.array:
        pass

class ParserSony(ParserTIF):
    def __init__(self):
        super().__init__()

    def TIF_STRIP(self, ptr, data):
        width = data["ImageWidth"]
        height = data["ImageLength"]
        bits = data["BitsPerSample"]
        offset = data["StripOffsets"]
        row = data["RowsPerStrip"]

        ptr.seek(offset)
        size = width*height

        raw = struct.unpack("H"*size, ptr.read(2*size))
        raw = np.asarray(raw, dtype=np.uint16).reshape((height, width))

        img = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2RGB)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        img = img * np.asarray([2048, 1024, 2128])
        img = cv2.normalize(img, None, 0, 2**16, cv2.NORM_MINMAX).astype(np.uint16)

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def read(self, path: str) -> np.array:
        file = open(path, 'rb')

        _, _, off = self.TIF_HEAD(file)
        while off > 0:
            tag, off = self.TIF_IFD(file, off)
            data = self.TIF_TAG_PARSE(file, tag)

            if "StripOffsets" in data:
                return self.TIF_STRIP(file, data)

            if "SubIFDs" in data:
                sub_tag, sub_off = self.TIF_IFD(file, data["SubIFDs"])
                sub_data = self.TIF_TAG_PARSE(file, sub_tag)
                if "StripOffsets" in sub_data:
                    return self.TIF_STRIP(file, sub_data)

if __name__ == "__main__":
    img1 = ParserRAW().read("toast.ARW")
    img2 = ParserSony().read("toast.ARW")

    img1 = cv2.resize(img1, None, fx=0.25, fy=0.25)
    img2 = cv2.resize(img2, None, fx=0.25, fy=0.25)

    cv2.imshow("img1", img1)
    cv2.imshow("img2", img2)
    cv2.waitKey()
