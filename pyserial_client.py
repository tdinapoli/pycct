"""
Minimal pyserial helper that mirrors the SCPI-style command construction used by the
Jeti board driver and routes traffic over a serial port.
"""

import datetime
import logging
import struct
from enum import Enum
from typing import Iterable, Optional
import re

import serial

from commands import (
    CalculateSubCommand,
    CommandCategory,
    ControlCommand,
    FetchCommand,
    FetchSubCommand,
    GeneralCommand,
    HelpCommand,
    MeasureCommand,
    MemoryCommand,
    ParameterCommand,
    ParameterSubCommand,
    StatusCommand,
)

LOGGER = logging.getLogger(__name__)


class CommandFactory:
    """Recreate the prefixes/formatters that the C# driver exposes via CategorySCPI."""

    @staticmethod
    def build_command(
        category: CommandCategory,
        command: str,
        *,
        is_getter: bool = True,
        args: Optional[Iterable[str]] = None,
        subcommand: Optional[str] = None,
    ) -> str:
        prefix = category.value
        payload = command
        if subcommand:
            payload = f"{payload}:{subcommand}"
        if is_getter:
            payload = f"{payload}?"
        elif args:
            payload = f"{payload} {' '.join(args)}"
        return f"*{prefix}{payload}"


class PySerialJetiClient:
    """Wraps pyserial to send commands and convert responses to Python types."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 1.0,
        line_ending: str = "\r\n",
    ):
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self._line_ending = line_ending.encode("ascii")
        self._factory = CommandFactory()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._serial.close()

    def set_timeout(self, timeout: float) -> None:
        self._serial.timeout = timeout

    def write_command(self, command: str) -> None:
        LOGGER.debug("Sending: %s", command)
        self._serial.write(command.encode("ascii") + self._line_ending)
        self._serial.flush()

    def read_text_line(self) -> str:
        raw = self._serial.readline()
        text = raw.decode(errors="ignore").strip()
        LOGGER.debug("Received text: %s", text)
        return text

    def request_line(self, command: str) -> str:
        self.write_command(command)
        return self.read_text_line()

    def request_binary(
        self, command: str, *, expected_length: Optional[int] = None
    ) -> bytes:
        self.write_command(command)
        if expected_length is not None:
            data = self._serial.read(expected_length)
        else:
            data = self._serial.read_until(self._line_ending)
        LOGGER.debug("Received binary (%d bytes)", len(data))
        return data

    def request_single_line(self, command: str) -> str:
        return self.request_line(command)

    def request_float_value(self, command: str) -> float:
        payload = self.request_single_line(command)
        try:
            return float(parse_float_value(payload))
        except ValueError as exc:
            raise ValueError(
                f"expected numeric response for {command!r}, got {payload!r}"
            ) from exc

    def request_multi_line(self, command: str) -> list[str]:
        self.write_command(command)
        lines = []
        while True:
            line = self.read_text_line()
            if not line:
                break
            lines.append(line)
        return lines

    def request_binary_confirmation(self, command: str) -> bool:
        value = self.request_single_line(command)
        return value.strip().upper() in {"1", "TRUE", "ON", "YES", "OK"}

    def request_binary_word(
        self,
        command: str,
        *,
        expected_length: int = 2,
        byte_order: str = "little",
        signed: bool = False,
    ) -> int:
        data = self.request_binary(command, expected_length=expected_length)
        if len(data) < expected_length:
            raise ValueError("binary word shorter than expected")
        return int.from_bytes(data[:expected_length], byte_order, signed=signed)

    def _build_command(
        self,
        category: CommandCategory,
        command: str,
        *,
        is_getter: bool = True,
        args: Iterable[str] | None = None,
        subcommand: str | Enum | None = None,
    ) -> str:
        sub = (
            subcommand.value
            if isinstance(subcommand, Enum)
            else str(subcommand)
            if subcommand is not None
            else None
        )
        return self._factory.build_command(
            category,
            command,
            is_getter=is_getter,
            args=args,
            subcommand=sub,
        )

    def _prepare_parameter_command(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
        is_getter: bool = True,
    ) -> str:
        command_name = command.value
        normalized_args = [str(argument) for argument in args] if args is not None else []
        sub_arg: Optional[str] = None
        if subcommand and subcommand != ParameterSubCommand.NONE:
            if subcommand == ParameterSubCommand.POLY:
                if not normalized_args:
                    raise ValueError("Poly parameter requires an appended index argument.")
                suffix = normalized_args.pop(0)
                command_name = f"{command_name}{suffix}"
            else:
                sub_arg = subcommand.value
        return self._build_command(
            CommandCategory.PARAMETER,
            command_name,
            is_getter=is_getter,
            args=normalized_args or None,
            subcommand=sub_arg,
        )

    def parameter_string(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> str:
        payload = self._prepare_parameter_command(
            command, subcommand=subcommand, args=args, is_getter=True
        )
        return self.request_single_line(payload)

    def parameter_float(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> float:
        payload = self._prepare_parameter_command(
            command, subcommand=subcommand, args=args, is_getter=True
        )
        return self.request_float_value(payload)

    def parameter_bool(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> bool:
        payload = self._prepare_parameter_command(
            command, subcommand=subcommand, args=args, is_getter=True
        )
        return self.request_binary_confirmation(payload)

    def parameter_lines(
        self,
        command: ParameterCommand,
        *,
        subcommand: ParameterSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> list[str]:
        payload = self._prepare_parameter_command(
            command, subcommand=subcommand, args=args, is_getter=True
        )
        return self.request_multi_line(payload)

    def write_parameter(
        self,
        command: ParameterCommand,
        values: Iterable[str],
        *,
        subcommand: ParameterSubCommand | None = None,
    ) -> bool:
        payload = self._prepare_parameter_command(
            command, subcommand=subcommand, args=values, is_getter=False
        )
        return self.request_binary_confirmation(payload)

    def general_string(
        self,
        command: GeneralCommand,
        *,
        args: Iterable[str] | None = None,
        is_getter: bool = True,
    ) -> str | None:
        payload = self._build_command(
            CommandCategory.GENERAL, command, is_getter=is_getter, args=args
        )
        if not is_getter and command == GeneralCommand.USBRST:
            self.request_line(payload)
            return None
        if not is_getter:
            return str(self.request_binary_confirmation(payload))
        return self.request_single_line(payload)

    def fetch_string(
        self,
        command: FetchCommand,
        *,
        subcommand: FetchSubCommand | None = None,
        args: Iterable[str] | None = None,
    ) -> str:
        payload = self._build_command(
            CommandCategory.FETCH,
            command,
            is_getter=True,
            args=args,
            subcommand=subcommand,
        )
        return self.request_single_line(payload)

    def measure_word(
        self,
        command: MeasureCommand,
        *,
        args: Iterable[str] | None = None,
    ) -> int:
        payload = self._build_command(
            CommandCategory.MEASURE,
            command,
            is_getter=False,
            args=args,
        )
        return self.request_binary_word(payload)

    def status_string(self, command: StatusCommand) -> str:
        payload = self._build_command(CommandCategory.STATUS, command, is_getter=True)
        return self.request_single_line(payload)

    def help_string(self, command: HelpCommand) -> str:
        payload = self._build_command(CommandCategory.HELP, command, is_getter=True)
        return self.request_single_line(payload)


def parse_float_value(msg: str):
    try:
        result = re.findall(r"[0-9]+\.?[0-9]*|[0-9]*\.?[0-9]+", msg)
        print(result)
        if len(result) == 0:
            raise ValueError(
                f"Expected to find a floating point number, instead found '{msg}'"
            )
        return result[0]
    except Exception as e:
        raise ValueError(
            f"Parsing message '{msg}' for a floating point number raise exception {e}"
        )
