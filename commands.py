from enum import Enum, auto


class _StrEnum(str, Enum):
    def _generate_next_value_(self, name: str, start: int, count: int, last_values: list) -> str:  # type: ignore[override]
        return name


class CommandCategory(_StrEnum):
    GENERAL = ""
    PARAMETER = "PARA:"
    CONFIGURE = "CONF:"
    CONTROL = "CONTR:"
    CALCULATE = "CALC:"
    CALIBRATE = "CALIB:"
    MEMORY = "MMEM:"
    MEASURE = "MEAS:"
    FETCH = "FETCH:"
    STATUS = "STAT:"
    HELP = ""


class GeneralCommand(_StrEnum):
    BOOT = auto()
    IDN = auto()
    RST = auto()
    VERS = auto()
    SERN = auto()
    SPNUM = auto()
    DEVNUM = auto()
    USBRST = auto()
    BAUD = auto()


class ParameterCommand(_StrEnum):
    ADCR = auto()
    ADCV = auto()
    ALLPARA = auto()
    ANCFAC = auto()
    AVER = auto()
    BACKUP = auto()
    BAUD = auto()
    BOXCA = auto()
    CALIBN = auto()
    DATE = auto()
    DEVNUM = auto()
    DIRECT = auto()
    ETH = auto()
    FAULTPI = auto()
    FAST = auto()
    FIT = auto()
    FLAS = auto()
    FORM = auto()
    FUNC = auto()
    GAIN = auto()
    GPIO = auto()
    LAMPE = auto()
    LAMPP = auto()
    LASERINT = auto()
    LASERLIM = auto()
    MAXTIN = auto()
    MAXAVER = auto()
    OFFCORR = auto()
    OFFS = auto()
    OVSAMP = auto()
    PDAG = auto()
    PIXBIN = auto()
    PIXEL = auto()
    PIXRAN = auto()
    PONTIM = auto()
    PRESC = auto()
    RESTORE = auto()
    SAVE = auto()
    SDEL = auto()
    SENS = auto()
    SERN = auto()
    SYNCFREQ = auto()
    SYNCMOD = auto()
    SPNUM = auto()
    TEMPC = auto()
    TIME = auto()
    TINT = auto()
    TRIG = auto()
    TRSL = auto()
    WRAN = auto()
    TROUTDEL = auto()
    MOTOR = auto()
    DEFAULT = auto()


class ParameterSubCommand(Enum):
    NONE = ""
    POLY = "Poly"
    IPADDR = "IPADDR"
    GWADDR = "GWADDR"
    SNMASK = "SNMASK"
    MACADDR = "MACADDR"
    OPENDELAY = "OPENDELay"
    CLOSEDELAY = "CLOSEDELay"
    DHCP = "DHCP"


class ConfigureCommand(_StrEnum):
    BATTEN = auto()
    BTEN = auto()
    DISPEN = auto()
    ETHEN = auto()
    PIEZOEN = auto()
    RTCEN = auto()
    SDCARDEN = auto()
    MINTIN = auto()


class ControlCommand(_StrEnum):
    GPIO = auto()
    LAMP = auto()
    LASER = auto()
    TIMESTAMP = auto()
    MOTOR = auto()
    CONNTYPE = auto()
    LED = auto()


class CalculateCommand(_StrEnum):
    CCT = auto()
    CHROMUV = auto()
    CHROMXY = auto()
    CRI = auto()
    DARK = auto()
    DWLPE = auto()
    LIGHT = auto()
    PHOTO = auto()
    RADIO = auto()
    REFER = auto()
    RGB = auto()
    SPRAD = auto()


class CalculateSubCommand(Enum):
    NONE = ""
    WAV = "WAV"
    PIX = "PIX"


class CalibrateCommand(_StrEnum):
    DATA = auto()
    DEL = auto()
    STAT = auto()


class MemoryCommand(_StrEnum):
    CAT = auto()
    COPY = auto()
    DATA = auto()
    DEL = auto()
    MOVE = auto()
    CLEAR = auto()


class MeasureCommand(_StrEnum):
    BATT = auto()
    DARK = auto()
    FLIC = auto()
    LIGHT = auto()
    REFE = auto()
    TEMPE = auto()
    TIADAPT = auto()
    DARKtext = auto()
    LIGHTtext = auto()
    RAW = auto()


class FetchCommand(_StrEnum):
    AVER = auto()
    DARK = auto()
    LEVEL = auto()
    LIGHT = auto()
    REFE = auto()
    TINT = auto()


class FetchSubCommand(Enum):
    NONE = ""
    ADAPT = "ADAPT"
    DARK = "DARK"
    LAST = "LAST"
    LIGHT = "LIGHT"
    REFER = "REFER"


class StatusCommand(_StrEnum):
    ENQU = auto()
    ERR = auto()
    TXTERR = auto()


class HelpCommand(_StrEnum):
    HELP = "HELP"
    PARA = "PARA"
    CONF = "CONF"
    CONTR = "CONTR"
    CALC = "CALC"
    CALIB = "CALIB"
    MMEM = "MMEM"
    MEAS = "MEAS"
    FETCH = "FETCH"
