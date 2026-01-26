"""
StringCowboy - Prepend and/or append text to all strings in a list.

Lassos each string in a list and brands them with prefix and suffix text.
"""

MAX_INPUTS = 20


class StringCowboy:
    """Prepend and/or append text to all strings in a list."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {
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
        }
        for i in range(1, MAX_INPUTS + 1):
            optional[f"string_{i}"] = ("STRING", {"forceInput": True})

        return {
            "required": {
                "mode": (["prepend", "append", "both"],),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "wrangle"
    CATEGORY = "Trent/Text"

    def wrangle(self, mode, prefix="", suffix="", **kwargs):
        # Collect all connected string inputs in order
        strings = []
        for i in range(1, MAX_INPUTS + 1):
            key = f"string_{i}"
            if key in kwargs and kwargs[key] is not None:
                val = kwargs[key]
                if isinstance(val, list):
                    strings.extend(val)
                else:
                    strings.append(str(val))

        # Apply prefix/suffix based on mode
        result = []
        for text in strings:
            text = str(text)
            if mode == "prepend":
                text = prefix + text
            elif mode == "append":
                text = text + suffix
            elif mode == "both":
                text = prefix + text + suffix
            result.append(text)

        return (result,)


NODE_CLASS_MAPPINGS = {
    "StringCowboy": StringCowboy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringCowboy": "String Cowboy",
}
