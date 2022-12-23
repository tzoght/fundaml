import math
import struct
import numpy as np

def binary(number: float):
    """Print the mantissa and exponent of a float.

    Parameters
    ----------
    number : float
        A floating point number

    Returns
    -------
    None : NoneType
        Returns nothing, prints output.

    Examples
    --------
    >>> binary(1.25)
     Decimal: 1.25 x 2^0
      Binary: 1.01 x 2^0

        Sign: 0 (+)
    Mantissa: 01 (0.25)
    Exponent: 0 (0)
    """

    # 1. The struct module packs a float into a string of bytes.
    packed = struct.pack("!d", float(number))
    # 2. The bytes are converted to integers.
    integers = [c for c in packed]
    # 3. The integers are converted to binary strings.
    binaries = [bin(i) for i in integers]
    # 4. The "0b" prefix is stripped from the binary strings.
    stripped_binaries = [s.replace("0b", "") for s in binaries]
    # 5. The binary strings are padded with leading zeros to ensure they are 8 characters long.
    padded = [s.rjust(8, "0") for s in stripped_binaries]
    # 6. The padded binary strings are concatenated into one giant string.
    final = "".join(padded)
    assert len(final) == 64, "something went wrong..."
    # 7. The giant string is split into 3 parts: the sign, the exponent, and the mantissa.
    sign, exponent_plus_1023, mantissa = final[0], final[1:12], final[12:]
    sign_str = "" if int(sign) == 0 else "-"
    mantissa_base10 = (
        int(mantissa, 2) / 2 ** 52
    )  
    mantissa_base10_str = str(mantissa_base10)[2:] 
    # 8. The sign is converted from binary to decimal.
    mantissa_index = mantissa.rfind("1")
    # 9. The mantissa is converted from binary to decimal.
    mantissa_index = 0 if mantissa_index == -1 else mantissa_index
    # 10. The exponent is converted from binary to decimal.
    exponent_base10 = int(exponent_plus_1023, 2) - 1023
    print(f"\n - Decimal: {sign_str}1.{mantissa_base10_str} x 2^{exponent_base10}")
    print(
        f" - Binary: {sign_str}1.{mantissa[:mantissa_index + 1]} x 2^{exponent_base10:b}"
    )
    print()
    print(f"- Sign: {sign} ({'+' if sign == '0' else '-'})")
    print(f"- Mantissa: {mantissa[:mantissa_index + 1]} ({mantissa_base10})")
    print(f"- Exponent: {exponent_base10:b} ({exponent_base10})\n")


def calc_spacing(number: float):
    """ Calculate the spacing between two floating point numbers.
    """
    return np.nextafter(number, 2 * number) - number
