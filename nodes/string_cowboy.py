"""
StringListCowboy - Make a list of strings with optional prepend/append.

Lassos strings together into a list and brands them with prefix/suffix text.
Works like Impact Pack's MakeAnyList but specialized for strings.
"""


class AnyType(str):
    """Wildcard type that matches any input type."""

    def __ne__(self, __value: object) -> bool:
        return False


any_typ = AnyType("*")


class StringListCowboy:
    """Make a list of strings with optional prepend/append to each."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "prefix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to add before each string"
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Text to add after each string"
                }),
                "value1": (any_typ,),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "wrangle"
    CATEGORY = "Trent/Text"

    def wrangle(self, prefix="", suffix="", **kwargs):
        # Collect all connected value inputs
        strings = []
        for key, val in kwargs.items():
            if key.startswith("value") and val is not None:
                if isinstance(val, list):
                    strings.extend(str(v) for v in val)
                else:
                    strings.append(str(val))

        # Apply prefix/suffix to each string
        result = []
        for text in strings:
            result.append(f"{prefix}{text}{suffix}")

        return (result,)


NODE_CLASS_MAPPINGS = {
    "StringListCowboy": StringListCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringListCowboy": "String List Cowboy",
}
