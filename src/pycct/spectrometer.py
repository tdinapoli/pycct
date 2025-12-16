import struct
import time
from enum import Enum
from typing import Any, Iterable, Self, Sequence
from dataclasses import dataclass
import itertools

import numpy as np
from numpy.typing import NDArray

from .commands import (
    CalculateCommand,
    CalculateSubCommand,
    CalibrateCommand,
    CommandCategory,
    ConfigureCommand,
    ControlCommand,
    FetchCommand,
    FetchSubCommand,
    GeneralCommand,
    HelpCommand,
    MeasureCommand,
    MeasureRawDataFormat,
    MemoryCommand,
    ParameterCommand,
    ParameterSubCommand,
    StatusCommand,
)
from .pyserial_client import PySerialJetiClient

_FLOAT_RESPONSE_PARAMETERS = {
    ParameterCommand.ADCR,
    ParameterCommand.AVER,
    ParameterCommand.FIT,
    ParameterCommand.GAIN,
    ParameterCommand.OFFS,
    ParameterCommand.PIXEL,
    ParameterCommand.TINT,
    ParameterCommand.TRIG,
    ParameterCommand.TRSL,
    ParameterCommand.TROUTDEL,
}

_MULTI_LINE_RESPONSE_PARAMETERS = {ParameterCommand.ALLPARA}

_BOOLEAN_RESPONSE_PARAMETERS = {ParameterCommand.LAMPE}


@dataclass(frozen=True)
class SpectrometerParams:
    wavelengths: np.ndarray
    fit_constants: Sequence[float]
    pixels: int
    amplitude_correction: NDArray[np.float64]
    vers: str
    model: str
    devnum: str
    gain: float
    offset_mv: float
    adc_resolution: int


class Spectrometer:
    """SCPI wrapper for the Jeti-based compact spectrometer."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        timeout: float = 1.0,
        line_ending: str = "\r\n",
    ):
        self._client = PySerialJetiClient(
            port=port, baudrate=baudrate, timeout=timeout, line_ending=line_ending
        )

        # TODO: enclose this behavior and data in a configuration class/method
        self._client.write_command(
            self._build_command(CommandCategory.PARAMETER, ParameterCommand.PIXEL)
        )
        self._n_pixels = int(self._client.read_text_line())

    def __enter__(self) -> Self:
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._client.__exit__(exc_type, exc_value, traceback)

    def set_timeout(self, timeout: float) -> None:
        """Expose the embedded serial timeout for long-running acquisitions."""
        self._client.set_timeout(timeout)

    def _build_command(
        self,
        category: CommandCategory,
        command: str | Enum,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
        subcommand: str | Enum | None = None,
    ) -> str:
        """Generate the ASCII command string for a given category."""

        command_name = command.value if isinstance(command, Enum) else str(command)
        normalized_args = (
            [str(argument) for argument in args] if args is not None else None
        )
        return self._client._build_command(
            category,
            command_name,
            is_getter=is_getter,
            args=normalized_args,
            subcommand=subcommand,
        )

    def general(
        self,
        command: GeneralCommand,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str | None:
        """Low-level wrapper for GENERAL category commands."""
        return self._client.general_string(command, args=args, is_getter=is_getter)

    def parameter(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> Any:
        """Low-level wrapper for PARAMETER category commands."""
        if is_getter:
            return self.read_parameter(command, subcommand=subcommand, args=args)
        return self.write_parameter(command, args or [], subcommand=subcommand)

    def configure(
        self,
        command: ConfigureCommand,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str | bool:
        """Low-level wrapper for CONFIGURE category commands."""
        payload = self._build_command(
            CommandCategory.CONFIGURE, command, is_getter=is_getter, args=args
        )
        if is_getter:
            return self._client.request_single_line(payload)
        return self._client.request_binary_confirmation(payload)

    def control(
        self,
        command: ControlCommand,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str | bool:
        """Low-level wrapper for CONTROL category commands."""
        payload = self._build_command(
            CommandCategory.CONTROL, command, is_getter=is_getter, args=args
        )
        if is_getter:
            return self._client.request_single_line(payload)
        return self._client.request_binary_confirmation(payload)

    def calculate(
        self,
        command: CalculateCommand,
        *,
        subcommand: CalculateSubCommand | None = None,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str | bool:
        """Low-level wrapper for CALCULATE category commands."""
        payload = self._build_command(
            CommandCategory.CALCULATE,
            command,
            is_getter=is_getter,
            args=args,
            subcommand=(
                subcommand.value
                if subcommand is not None and subcommand != CalculateSubCommand.NONE
                else None
            ),
        )
        if is_getter:
            return self._client.request_single_line(payload)
        return self._client.request_binary_confirmation(payload)

    def calibrate(
        self,
        command: CalibrateCommand,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str | bool:
        """Low-level wrapper for CALIBRATE category commands."""
        payload = self._build_command(
            CommandCategory.CALIBRATE, command, is_getter=is_getter, args=args
        )
        if is_getter:
            return self._client.request_single_line(payload)
        return self._client.request_binary_confirmation(payload)

    def memory(
        self,
        command: MemoryCommand,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str | bool:
        """Low-level wrapper for MEMORY category commands."""
        payload = self._build_command(
            CommandCategory.MEMORY, command, is_getter=is_getter, args=args
        )
        if is_getter:
            return self._client.request_single_line(payload)
        return self._client.request_binary_confirmation(payload)

    def measure(
        self,
        command: MeasureCommand,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str | int:
        """Low-level wrapper for MEASURE category commands."""
        payload = self._build_command(
            CommandCategory.MEASURE, command, is_getter=is_getter, args=args
        )
        if is_getter:
            return self._client.request_single_line(payload)
        return self._client.request_binary_word(payload)

    def fetch(
        self,
        command: FetchCommand,
        *,
        subcommand: FetchSubCommand | None = None,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str:
        """Low-level wrapper for FETCH category commands."""
        payload = self._build_command(
            CommandCategory.FETCH,
            command,
            is_getter=is_getter,
            args=args,
            subcommand=(
                subcommand.value
                if subcommand is not None and subcommand != FetchSubCommand.NONE
                else None
            ),
        )
        return self._client.request_single_line(payload)

    def status(
        self,
        command: StatusCommand,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str:
        """Low-level wrapper for STATUS category commands."""
        payload = self._build_command(
            CommandCategory.STATUS, command, is_getter=is_getter, args=args
        )
        return self._client.request_single_line(payload)

    def help_command(
        self,
        command: HelpCommand,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
    ) -> str:
        """Low-level wrapper for HELP category commands."""
        payload = self._build_command(
            CommandCategory.HELP, command, is_getter=is_getter, args=args
        )
        return self._client.request_single_line(payload)

    def identify(self) -> Any:
        """Return the device identification string that includes model/vendor info."""
        return self.general(GeneralCommand.IDN)

    def firmware_version(self) -> Any:
        """Return the firmware version currently running on the spectrometer."""
        return self.general(GeneralCommand.VERS)

    def serial_number(self) -> Any:
        """Return the device serial number reported by the spectrometer."""
        return self.general(GeneralCommand.SERN)

    def reset_device(self) -> Any:
        """Trigger a software reset on the spectrometer."""
        return self.general(GeneralCommand.RST, is_getter=False)

    def usb_reset(self) -> Any:
        """Issue a USB reset cycle for the connected spectrometer."""
        return self.general(GeneralCommand.USBRST, is_getter=False)

    def read_parameter(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> str | float | bool | list[str]:
        """Read back a parameter value using a user-facing name."""
        if command in _FLOAT_RESPONSE_PARAMETERS:
            return self.read_parameter_float(command, subcommand=subcommand, args=args)
        if command in _BOOLEAN_RESPONSE_PARAMETERS:
            return self.read_parameter_bool(command, subcommand=subcommand, args=args)
        if command in _MULTI_LINE_RESPONSE_PARAMETERS:
            return self.read_parameter_lines(command, subcommand=subcommand, args=args)
        return self.read_parameter_str(command, subcommand=subcommand, args=args)

    def write_parameter(
        self,
        command: ParameterCommand,
        values: str | Iterable[str],
        *,
        subcommand: ParameterSubCommand | None = None,
    ) -> bool:
        """Update a parameter with the provided value(s)."""
        if isinstance(values, str):
            normalized_values = [values]
        else:
            normalized_values = [str(value) for value in values]
        return self._client.write_parameter(
            command,
            subcommand=subcommand,
            values=normalized_values,
        )

    def write_parameter_bool(
        self,
        command: ParameterCommand,
        value: bool,
        *,
        subcommand: ParameterSubCommand | None = None,
    ) -> bool:
        """Convenience helper to write boolean parameters via the shared writer."""
        return self.write_parameter(
            command, "1" if value else "0", subcommand=subcommand
        )

    def read_parameter_str(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> str:
        """Read a simple string parameter."""
        return self._client.parameter_string(
            command,
            subcommand=subcommand,
            args=args,
        )

    def read_parameter_float(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> float:
        """Read a float-valued parameter."""
        return self._client.parameter_float(
            command,
            subcommand=subcommand,
            args=args,
        )

    def read_parameter_bool(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> bool:
        """Read a parameter that resolves to a boolean confirmation."""
        return self._client.parameter_bool(
            command,
            subcommand=subcommand,
            args=args,
        )

    def read_parameter_lines(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> list[str]:
        """Read a parameter that returns multiple text lines."""
        return self._client.parameter_lines(
            command,
            subcommand=subcommand,
            args=args,
        )

    def read_params(self) -> SpectrometerParams:
        wavelengths = self.compute_wavelenghts()
        fit_constants = self.get_fit_constants()
        pixels = self.read_parameter(ParameterCommand.PIXEL)
        vers = self.get_vers()
        model = self.read_parameter(ParameterCommand.SPNUM)
        devnum = self.read_parameter(ParameterCommand.DEVNUM)
        gain = self.read_parameter(ParameterCommand.GAIN)
        offset_mv = self.read_parameter(ParameterCommand.OFFS)
        adc_resolution = self.read_parameter(ParameterCommand.ADCR)
        amplitude_correction = self.get_amplitude_correction()
        return SpectrometerParams(
            wavelengths=wavelengths,
            fit_constants=fit_constants,
            pixels=pixels,
            vers=vers,
            model=model,
            devnum=devnum,
            gain=gain,
            offset_mv=offset_mv,
            adc_resolution=adc_resolution,
            amplitude_correction=amplitude_correction,
        )

    def compute_wavelenghts(self) -> list[float]:
        coefs = []
        for i in range(5):
            self._client.write_command(f"*PARA:FIT{i}?")
            text_line = self._client.read_text_line()
            coefs.append(float(text_line))
        pixels = list(range(self._n_pixels))
        result = list(np.polyval(coefs[::-1], pixels))
        self._client._serial.reset_input_buffer()
        return result

    def acquire_single_spectrum(
        self,
        tint: float,
        averages: int,
        data_format: MeasureRawDataFormat = MeasureRawDataFormat.SHORTS,
    ) -> tuple[int, list[float]]:
        """Acquire single spectrum. Returns a timestamp in microseconds and a list of counts"""
        command = self._client._build_command(
            CommandCategory.MEASURE,
            MeasureCommand.RAW.value,
            args=(str(tint), str(averages), str(data_format.value)),
            is_getter=False,
        )
        self._client.write_command(command)
        time.sleep(tint + tint * 1e-5)

        header = self._client._serial.read(4)
        timestamp = self._client._serial.read(4)
        footer = self._client._serial.read(4)
        if len(header + timestamp + footer) != 12:
            raise RuntimeError("could not properly read bytes before data")
        timestamp = int(struct.unpack("I", timestamp)[0])
        result = (timestamp, self._decode_spectrum(data_format))
        self._client._serial.reset_input_buffer()
        return result

    def _decode_spectrum(self, data_format: MeasureRawDataFormat) -> list[float]:
        match data_format:
            case MeasureRawDataFormat.SHORTS:
                return self._decode_spectrum_shorts()
            case MeasureRawDataFormat.ASCII:
                return self._decode_spectrum_ascii()
            case MeasureRawDataFormat.WL_SHORTS:
                return self._decode_spectrum_wl_shorts()

    def _decode_spectrum_shorts(self) -> list[float]:
        raw_bytes = self._client._serial.read(self._n_pixels * struct.calcsize("h"))
        return list(
            struct.unpack(
                f"{(self._n_pixels - 1)}hh",
                raw_bytes,  # TODO: fix this
            )
        )

    def _decode_spectrum_ascii(self) -> list[float]:
        raise NotImplementedError

    def _decode_spectrum_wl_shorts(self) -> list[float]:
        raise NotImplementedError

    def get_fit_constants(self) -> Sequence[float]:
        results = []
        try:
            for i in itertools.count():
                # TODO: change commands api to do this properly
                self._client.write_command(f"*PARA:FIT{i}?")
                result = float(self._client.read_text_line())
                results.append(result)
        except ValueError:
            pass
        return results

    def get_vers(self) -> str:
        self._client.write_command(
            self._client._build_command(CommandCategory.GENERAL, GeneralCommand.VERS)
        )
        return self._client.read_text_line()

    def get_amplitude_correction(self) -> NDArray[np.float64]:
        # TODO: implement this
        return np.array([])

    def measure_light(self) -> int:
        """Perform the LIGHT measurement sequence."""
        return self.measure(MeasureCommand.LIGHT, is_getter=False)

    def measure_dark(self) -> int:
        """Perform the DARK measurement sequence."""
        return self.measure(MeasureCommand.DARK, is_getter=False)

    def fetch_light(
        self,
        *,
        subcommand: FetchSubCommand | None = None,
    ) -> str:
        """Fetch the latest LIGHT data that is cached on the device."""
        return self.fetch(FetchCommand.LIGHT, subcommand=subcommand)

    def fetch_dark(
        self,
        *,
        subcommand: FetchSubCommand | None = None,
    ) -> str:
        """Fetch the latest DARK data that is cached on the device."""
        return self.fetch(FetchCommand.DARK, subcommand=subcommand)

    def read_status(
        self,
        *,
        command: StatusCommand = StatusCommand.ENQU,
    ) -> str:
        """Expose the main status query that most users care about."""
        return self.status(command)

    def get_integration_time(self) -> str:
        """Return the current integration time (TINT) from the spectrometer."""
        return self.read_parameter(ParameterCommand.TINT)

    def set_integration_time(self, microseconds: int | str) -> bool:
        """Set the spectrometer's integration time in microseconds."""
        return self.write_parameter(ParameterCommand.TINT, str(microseconds))

    def get_lamp_state(self) -> bool:
        """Return whether the internal lamp (LAMPE) is enabled."""
        return self.read_parameter_bool(ParameterCommand.LAMPE)

    def set_lamp_state(self, enabled: bool) -> bool:
        """Enable or disable the internal lamp."""
        return self.write_parameter_bool(ParameterCommand.LAMPE, enabled)

    def set_lamp_power(self, percent: int | str) -> bool:
        """Adjust the lamp power percentage (LAMPP)."""
        return self.write_parameter(ParameterCommand.LAMPP, str(percent))

    def get_adcr(self) -> float:
        """Return the current ADCR (Analog-to-Digital Conversion Range that configures the ADC scale) parameter."""
        return self.read_parameter(ParameterCommand.ADCR)

    def set_adcr(self, value: str | Iterable[str]) -> bool:
        """Set the ADCR (Analog-to-Digital Conversion Range that configures the ADC scale) parameter."""
        return self.write_parameter(ParameterCommand.ADCR, value)

    def get_adcv(self) -> str:
        """Return the current ADCV (Analog-to-Digital Conversion Voltage reference used for calibrations) parameter."""
        return self.read_parameter(ParameterCommand.ADCV)

    def set_adcv(self, value: str | Iterable[str]) -> bool:
        """Set the ADCV (Analog-to-Digital Conversion Voltage reference used for calibrations) parameter."""
        return self.write_parameter(ParameterCommand.ADCV, value)

    def get_allpara(self) -> list[str]:
        """Return the current ALLPARA (All Parameters dump that lists every writable setting) parameter."""
        return self.read_parameter(ParameterCommand.ALLPARA)

    def set_allpara(self, value: str | Iterable[str]) -> bool:
        """Set the ALLPARA (All Parameters dump that lists every writable setting) parameter."""
        return self.write_parameter(ParameterCommand.ALLPARA, value)

    def get_ancfac(self) -> str:
        """Return the current ANCFAC (Analog Correction Factor for fine tuning calibration curves) parameter."""
        return self.read_parameter(ParameterCommand.ANCFAC)

    def set_ancfac(self, value: str | Iterable[str]) -> bool:
        """Set the ANCFAC (Analog Correction Factor for fine tuning calibration curves) parameter."""
        return self.write_parameter(ParameterCommand.ANCFAC, value)

    def get_aver(self) -> float:
        """Return the current AVER (Averaging count used when the hardware computes on-board averages) parameter."""
        return self.read_parameter(ParameterCommand.AVER)

    def set_aver(self, value: str | Iterable[str]) -> bool:
        """Set the AVER (Averaging count used when the hardware computes on-board averages) parameter."""
        return self.write_parameter(ParameterCommand.AVER, value)

    def get_backup(self) -> str:
        """Return the current BACKUP (Backup configuration slot used to store factory defaults) parameter."""
        return self.read_parameter(ParameterCommand.BACKUP)

    def set_backup(self, value: str | Iterable[str]) -> bool:
        """Set the BACKUP (Backup configuration slot used to store factory defaults) parameter."""
        return self.write_parameter(ParameterCommand.BACKUP, value)

    def get_baud(self) -> str:
        """Return the current BAUD (Baud rate that governs the serial interface speed) parameter."""
        return self.read_parameter(ParameterCommand.BAUD)

    def set_baud(self, value: str | Iterable[str]) -> bool:
        """Set the BAUD (Baud rate that governs the serial interface speed) parameter."""
        return self.write_parameter(ParameterCommand.BAUD, value)

    def get_boxca(self) -> str:
        """Return the current BOXCA (Boxcar averaging window (the Carbox filter) applied in hardware smoothing) parameter."""
        return self.read_parameter(ParameterCommand.BOXCA)

    def set_boxca(self, value: str | Iterable[str]) -> bool:
        """Set the BOXCA (Boxcar averaging window (the Carbox filter) applied in hardware smoothing) parameter."""
        return self.write_parameter(ParameterCommand.BOXCA, value)

    def get_calibn(self) -> str:
        """Return the current CALIBN (Calibration Number selecting which stored calibration profile is active) parameter."""
        return self.read_parameter(ParameterCommand.CALIBN)

    def set_calibn(self, value: str | Iterable[str]) -> bool:
        """Set the CALIBN (Calibration Number selecting which stored calibration profile is active) parameter."""
        return self.write_parameter(ParameterCommand.CALIBN, value)

    def get_date(self) -> str:
        """Return the current DATE (Date stored on the device (typically YYYYMMDD)) parameter."""
        return self.read_parameter(ParameterCommand.DATE)

    def set_date(self, value: str | Iterable[str]) -> bool:
        """Set the DATE (Date stored on the device (typically YYYYMMDD)) parameter."""
        return self.write_parameter(ParameterCommand.DATE, value)

    def get_devnum(self) -> str:
        """Return the current DEVNUM (Device Number that uniquely identifies the spectrometer instance) parameter."""
        return self.read_parameter(ParameterCommand.DEVNUM)

    def set_devnum(self, value: str | Iterable[str]) -> bool:
        """Set the DEVNUM (Device Number that uniquely identifies the spectrometer instance) parameter."""
        return self.write_parameter(ParameterCommand.DEVNUM, value)

    def get_direct(self) -> str:
        """Return the current DIRECT (Direct operation mode that bypasses intermediate filtering) parameter."""
        return self.read_parameter(ParameterCommand.DIRECT)

    def set_direct(self, value: str | Iterable[str]) -> bool:
        """Set the DIRECT (Direct operation mode that bypasses intermediate filtering) parameter."""
        return self.write_parameter(ParameterCommand.DIRECT, value)

    def get_eth(self) -> str:
        """Return the current ETH (Ethernet configuration string (IP/Gateway/DNS) for the network interface) parameter."""
        return self.read_parameter(ParameterCommand.ETH)

    def set_eth(self, value: str | Iterable[str]) -> bool:
        """Set the ETH (Ethernet configuration string (IP/Gateway/DNS) for the network interface) parameter."""
        return self.write_parameter(ParameterCommand.ETH, value)

    def get_faultpi(self) -> str:
        """Return the current FAULTPI (Fault Protection Interlock flag that reflects a hardware fault state) parameter."""
        return self.read_parameter(ParameterCommand.FAULTPI)

    def set_faultpi(self, value: str | Iterable[str]) -> bool:
        """Set the FAULTPI (Fault Protection Interlock flag that reflects a hardware fault state) parameter."""
        return self.write_parameter(ParameterCommand.FAULTPI, value)

    def get_fast(self) -> str:
        """Return the current FAST (Fast acquisition flag that triggers expedited sampling paths) parameter."""
        return self.read_parameter(ParameterCommand.FAST)

    def set_fast(self, value: bool | str | Iterable[str]) -> bool:
        """
        Set the FAST (Fast acquisition flag that triggers expedited sampling paths) parameter.
        Accept booleans for the convenience helper that converts to bytes and fall back to general writers.
        """
        if isinstance(value, bool):
            return self.write_parameter_bool(ParameterCommand.FAST, value)
        return self.write_parameter(ParameterCommand.FAST, value)

    def get_fit(self) -> float:
        """Return the current FIT (Polynomial Fit coefficients that define the wavelength calibration) parameter."""
        return self.read_parameter(ParameterCommand.FIT)

    def set_fit(self, value: str | Iterable[str]) -> bool:
        """Set the FIT (Polynomial Fit coefficients that define the wavelength calibration) parameter."""
        return self.write_parameter(ParameterCommand.FIT, value)

    def get_flas(self) -> str:
        """Return the current FLAS (Flash memory control used for firmware and parameter storage) parameter."""
        return self.read_parameter(ParameterCommand.FLAS)

    def set_flas(self, value: str | Iterable[str]) -> bool:
        """Set the FLAS (Flash memory control used for firmware and parameter storage) parameter."""
        return self.write_parameter(ParameterCommand.FLAS, value)

    def get_form(self) -> str:
        """Return the current FORM (Output Format selection that dictates measurement formatting) parameter."""
        return self.read_parameter(ParameterCommand.FORM)

    def set_form(self, value: str | Iterable[str]) -> bool:
        """Set the FORM (Output Format selection that dictates measurement formatting) parameter."""
        return self.write_parameter(ParameterCommand.FORM, value)

    def get_func(self) -> str:
        """Return the current FUNC (Function selector that chooses the active measurement routine) parameter."""
        return self.read_parameter(ParameterCommand.FUNC)

    def set_func(self, value: str | Iterable[str]) -> bool:
        """Set the FUNC (Function selector that chooses the active measurement routine) parameter."""
        return self.write_parameter(ParameterCommand.FUNC, value)

    def get_gain(self) -> float:
        """Return the current GAIN (Analog Gain that amplifies the detector signal) parameter."""
        return self.read_parameter(ParameterCommand.GAIN)

    def set_gain(self, value: str | Iterable[str]) -> bool:
        """Set the GAIN (Analog Gain that amplifies the detector signal) parameter."""
        return self.write_parameter(ParameterCommand.GAIN, value)

    def get_gpio(self) -> str:
        """Return the current GPIO (General-Purpose Input/Output states exposed over the connector) parameter."""
        return self.read_parameter(ParameterCommand.GPIO)

    def set_gpio(self, value: str | Iterable[str]) -> bool:
        """Set the GPIO (General-Purpose Input/Output states exposed over the connector) parameter."""
        return self.write_parameter(ParameterCommand.GPIO, value)

    def get_lampe(self) -> bool:
        """Return the current LAMPE (Lamp Enable flag for switching the internal lamp on or off) parameter."""
        return self.read_parameter(ParameterCommand.LAMPE)

    def set_lampe(self, value: str | Iterable[str]) -> bool:
        """Set the LAMPE (Lamp Enable flag for switching the internal lamp on or off) parameter."""
        return self.write_parameter(ParameterCommand.LAMPE, value)

    def get_lampp(self) -> str:
        """Return the current LAMPP (Lamp Power level expressed as a percentage) parameter."""
        return self.read_parameter(ParameterCommand.LAMPP)

    def set_lampp(self, value: str | Iterable[str]) -> bool:
        """Set the LAMPP (Lamp Power level expressed as a percentage) parameter."""
        return self.write_parameter(ParameterCommand.LAMPP, value)

    def get_laserint(self) -> str:
        """Return the current LASERINT (Laser Intensity control for the integrated laser source) parameter."""
        return self.read_parameter(ParameterCommand.LASERINT)

    def set_laserint(self, value: str | Iterable[str]) -> bool:
        """Set the LASERINT (Laser Intensity control for the integrated laser source) parameter."""
        return self.write_parameter(ParameterCommand.LASERINT, value)

    def get_laserlim(self) -> str:
        """Return the current LASERLIM (Laser Limit ceiling to protect the optical source) parameter."""
        return self.read_parameter(ParameterCommand.LASERLIM)

    def set_laserlim(self, value: str | Iterable[str]) -> bool:
        """Set the LASERLIM (Laser Limit ceiling to protect the optical source) parameter."""
        return self.write_parameter(ParameterCommand.LASERLIM, value)

    def get_maxtin(self) -> str:
        """Return the current MAXTIN (Maximum Integration Time allowed by the hardware) parameter."""
        return self.read_parameter(ParameterCommand.MAXTIN)

    def set_maxtin(self, value: str | Iterable[str]) -> bool:
        """Set the MAXTIN (Maximum Integration Time allowed by the hardware) parameter."""
        return self.write_parameter(ParameterCommand.MAXTIN, value)

    def get_maxaver(self) -> str:
        """Return the current MAXAVER (Maximum Averaging count permitted by the board) parameter."""
        return self.read_parameter(ParameterCommand.MAXAVER)

    def set_maxaver(self, value: str | Iterable[str]) -> bool:
        """Set the MAXAVER (Maximum Averaging count permitted by the board) parameter."""
        return self.write_parameter(ParameterCommand.MAXAVER, value)

    def get_offcorr(self) -> str:
        """Return the current OFFCORR (Offset Correction coefficient applied to the detector baseline) parameter."""
        return self.read_parameter(ParameterCommand.OFFCORR)

    def set_offcorr(self, value: str | Iterable[str]) -> bool:
        """Set the OFFCORR (Offset Correction coefficient applied to the detector baseline) parameter."""
        return self.write_parameter(ParameterCommand.OFFCORR, value)

    def get_offs(self) -> float:
        """Return the current OFFS (Offsets array that specifies pixel-level offset corrections) parameter."""
        return self.read_parameter(ParameterCommand.OFFS)

    def set_offs(self, value: str | Iterable[str]) -> bool:
        """Set the OFFS (Offsets array that specifies pixel-level offset corrections) parameter."""
        return self.write_parameter(ParameterCommand.OFFS, value)

    def get_ovsamp(self) -> str:
        """Return the current OVSAMP (Oversampling multiplier that increases data resolution) parameter."""
        return self.read_parameter(ParameterCommand.OVSAMP)

    def set_ovsamp(self, value: str | Iterable[str]) -> bool:
        """Set the OVSAMP (Oversampling multiplier that increases data resolution) parameter."""
        return self.write_parameter(ParameterCommand.OVSAMP, value)

    def get_pdag(self) -> str:
        """Return the current PDAG (Photodiode Array Gain setting for the sensor readout) parameter."""
        return self.read_parameter(ParameterCommand.PDAG)

    def set_pdag(self, value: str | Iterable[str]) -> bool:
        """Set the PDAG (Photodiode Array Gain setting for the sensor readout) parameter."""
        return self.write_parameter(ParameterCommand.PDAG, value)

    def get_pixbin(self) -> str:
        """Return the current PIXBIN (Pixel Binning factor that groups adjacent pixels together) parameter."""
        return self.read_parameter(ParameterCommand.PIXBIN)

    def set_pixbin(self, value: str | Iterable[str]) -> bool:
        """Set the PIXBIN (Pixel Binning factor that groups adjacent pixels together) parameter."""
        return self.write_parameter(ParameterCommand.PIXBIN, value)

    def get_pixel(self) -> float:
        """Return the current PIXEL (Pixel index or value parameter exposed by the device) parameter."""
        return self.read_parameter(ParameterCommand.PIXEL)

    def set_pixel(self, value: str | Iterable[str]) -> bool:
        """Set the PIXEL (Pixel index or value parameter exposed by the device) parameter."""
        return self.write_parameter(ParameterCommand.PIXEL, value)

    def get_pixran(self) -> str:
        """Return the current PIXRAN (Pixel Range that limits which pixels are read out) parameter."""
        return self.read_parameter(ParameterCommand.PIXRAN)

    def set_pixran(self, value: str | Iterable[str]) -> bool:
        """Set the PIXRAN (Pixel Range that limits which pixels are read out) parameter."""
        return self.write_parameter(ParameterCommand.PIXRAN, value)

    def get_pontim(self) -> str:
        """Return the current PONTIM (Point Time (time per acquisition point) that sets timing granularity) parameter."""
        return self.read_parameter(ParameterCommand.PONTIM)

    def set_pontim(self, value: str | Iterable[str]) -> bool:
        """Set the PONTIM (Point Time (time per acquisition point) that sets timing granularity) parameter."""
        return self.write_parameter(ParameterCommand.PONTIM, value)

    def get_presc(self) -> str:
        """Return the current PRESC (Prescaler that divides the reference clock for timing) parameter."""
        return self.read_parameter(ParameterCommand.PRESC)

    def set_presc(self, value: str | Iterable[str]) -> bool:
        """Set the PRESC (Prescaler that divides the reference clock for timing) parameter."""
        return self.write_parameter(ParameterCommand.PRESC, value)

    def get_restore(self) -> str:
        """Return the current RESTORE (Restore flag that reverts to factory parameter defaults) parameter."""
        return self.read_parameter(ParameterCommand.RESTORE)

    def set_restore(self, value: str | Iterable[str]) -> bool:
        """Set the RESTORE (Restore flag that reverts to factory parameter defaults) parameter."""
        return self.write_parameter(ParameterCommand.RESTORE, value)

    def get_save(self) -> str:
        """Return the current SAVE (Save flag that writes the current configuration to flash) parameter."""
        return self.read_parameter(ParameterCommand.SAVE)

    def set_save(self, value: str | Iterable[str]) -> bool:
        """Set the SAVE (Save flag that writes the current configuration to flash) parameter."""
        return self.write_parameter(ParameterCommand.SAVE, value)

    def get_sdel(self) -> str:
        """Return the current SDEL (Sample Delay (SDEL) between trigger reception and readout start) parameter."""
        return self.read_parameter(ParameterCommand.SDEL)

    def set_sdel(self, value: str | Iterable[str]) -> bool:
        """Set the SDEL (Sample Delay (SDEL) between trigger reception and readout start) parameter."""
        return self.write_parameter(ParameterCommand.SDEL, value)

    def get_sens(self) -> str:
        """Return the current SENS (Sensitivity level that scales the detector response) parameter."""
        return self.read_parameter(ParameterCommand.SENS)

    def set_sens(self, value: str | Iterable[str]) -> bool:
        """Set the SENS (Sensitivity level that scales the detector response) parameter."""
        return self.write_parameter(ParameterCommand.SENS, value)

    def get_sern(self) -> str:
        """Return the current SERN (Serial Number string reported by the hardware) parameter."""
        return self.read_parameter(ParameterCommand.SERN)

    def set_sern(self, value: str | Iterable[str]) -> bool:
        """Set the SERN (Serial Number string reported by the hardware) parameter."""
        return self.write_parameter(ParameterCommand.SERN, value)

    def get_syncfreq(self) -> str:
        """Return the current SYNCFREQ (Synchronization Frequency used for external triggers) parameter."""
        return self.read_parameter(ParameterCommand.SYNCFREQ)

    def set_syncfreq(self, value: str | Iterable[str]) -> bool:
        """Set the SYNCFREQ (Synchronization Frequency used for external triggers) parameter."""
        return self.write_parameter(ParameterCommand.SYNCFREQ, value)

    def get_syncmod(self) -> str:
        """Return the current SYNCMOD (Synchronization Mode that defines how sync signals are handled) parameter."""
        return self.read_parameter(ParameterCommand.SYNCMOD)

    def set_syncmod(self, value: str | Iterable[str]) -> bool:
        """Set the SYNCMOD (Synchronization Mode that defines how sync signals are handled) parameter."""
        return self.write_parameter(ParameterCommand.SYNCMOD, value)

    def get_spnum(self) -> str:
        """Return the current SPNUM (Spectrum Number index used for referencing stored sweeps) parameter."""
        return self.read_parameter(ParameterCommand.SPNUM)

    def set_spnum(self, value: str | Iterable[str]) -> bool:
        """Set the SPNUM (Spectrum Number index used for referencing stored sweeps) parameter."""
        return self.write_parameter(ParameterCommand.SPNUM, value)

    def get_tempc(self) -> str:
        """Return the current TEMPC (Temperature Compensation coefficient affecting wavelength math) parameter."""
        return self.read_parameter(ParameterCommand.TEMPC)

    def set_tempc(self, value: str | Iterable[str]) -> bool:
        """Set the TEMPC (Temperature Compensation coefficient affecting wavelength math) parameter."""
        return self.write_parameter(ParameterCommand.TEMPC, value)

    def get_time(self) -> str:
        """Return the current TIME (Local Time-of-day clock maintained by the spectrometer) parameter."""
        return self.read_parameter(ParameterCommand.TIME)

    def set_time(self, value: str | Iterable[str]) -> bool:
        """Set the TIME (Local Time-of-day clock maintained by the spectrometer) parameter."""
        return self.write_parameter(ParameterCommand.TIME, value)

    def get_tint(self) -> float:
        """Return the current TINT (Integration Time that controls how long the sensor collects light) parameter."""
        return self.read_parameter(ParameterCommand.TINT)

    def set_tint(self, value: str | Iterable[str]) -> bool:
        """Set the TINT (Integration Time that controls how long the sensor collects light) parameter."""
        return self.write_parameter(ParameterCommand.TINT, value)

    def get_trig(self) -> float:
        """Return the current TRIG (Trigger configuration that determines when snapshots start) parameter."""
        return self.read_parameter(ParameterCommand.TRIG)

    def set_trig(self, value: str | Iterable[str]) -> bool:
        """Set the TRIG (Trigger configuration that determines when snapshots start) parameter."""
        return self.write_parameter(ParameterCommand.TRIG, value)

    def get_trsl(self) -> float:
        """Return the current TRSL (Trigger Signal Level threshold guarding external triggers) parameter."""
        return self.read_parameter(ParameterCommand.TRSL)

    def set_trsl(self, value: str | Iterable[str]) -> bool:
        """Set the TRSL (Trigger Signal Level threshold guarding external triggers) parameter."""
        return self.write_parameter(ParameterCommand.TRSL, value)

    def get_wran(self) -> str:
        """Return the current WRAN (Wavelength Range window that defines the scanned spectrum) parameter."""
        return self.read_parameter(ParameterCommand.WRAN)

    def set_wran(self, value: str | Iterable[str]) -> bool:
        """Set the WRAN (Wavelength Range window that defines the scanned spectrum) parameter."""
        return self.write_parameter(ParameterCommand.WRAN, value)

    def get_troutdel(self) -> float:
        """Return the current TROUTDEL (Trigger Output Delay before the board emits a trigger pulse) parameter."""
        return self.read_parameter(ParameterCommand.TROUTDEL)

    def set_troutdel(self, value: str | Iterable[str]) -> bool:
        """Set the TROUTDEL (Trigger Output Delay before the board emits a trigger pulse) parameter."""
        return self.write_parameter(ParameterCommand.TROUTDEL, value)

    def get_motor(self) -> str:
        """Return the current MOTOR (Motor control parameter (if the device exposes a motorized shutter)) parameter."""
        return self.read_parameter(ParameterCommand.MOTOR)

    def set_motor(self, value: str | Iterable[str]) -> bool:
        """Set the MOTOR (Motor control parameter (if the device exposes a motorized shutter)) parameter."""
        return self.write_parameter(ParameterCommand.MOTOR, value)

    def get_default(self) -> str:
        """Return the current DEFAULT (Default parameter set that reflects factory starting values) parameter."""
        return self.read_parameter(ParameterCommand.DEFAULT)

    def set_default(self, value: str | Iterable[str]) -> bool:
        """Set the DEFAULT (Default parameter set that reflects factory starting values) parameter."""
        return self.write_parameter(ParameterCommand.DEFAULT, value)
