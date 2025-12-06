#!/bin/bash
# Script to add custom colors to all Trent Nodes

cd ~/ComfyUI/custom_nodes/TrentNodes/nodes

# Your exact colors
BACKGROUND_COLOR='    BACKGROUND_COLOR = "#0a1218"  # Dark background'
FOREGROUND_COLOR='    FOREGROUND_COLOR = "#0c1b21"  # Darker teal'

echo "Adding custom colors to all nodes..."

# Function to add colors to a node file
add_colors_to_file() {
    local file=$1
    
    # Check if colors already exist
    if grep -q "BACKGROUND_COLOR" "$file"; then
        echo "  ⏭️  $file already has colors, skipping"
        return
    fi
    
    # Find the CATEGORY line and add colors after it
    if grep -q "CATEGORY = " "$file"; then
        # Use awk to insert after CATEGORY line
        awk -v bg="$BACKGROUND_COLOR" -v fg="$FOREGROUND_COLOR" '
        /CATEGORY = / {
            print
            print bg
            print fg
            next
        }
        {print}
        ' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
        echo "  ✅ Added colors to $file"
    else
        echo "  ⚠️  No CATEGORY found in $file, skipping"
    fi
}

# Process all Python files in nodes/
for file in *.py; do
    if [ "$file" != "__init__.py" ]; then
        add_colors_to_file "$file"
    fi
done

# Process wan_vace subfolder
if [ -f "wan_vace/wan_vace_keyframes.py" ]; then
    add_colors_to_file "wan_vace/wan_vace_keyframes.py"
fi

echo ""
echo "✅ Color theming complete!"
echo ""
echo "Colors applied:"
echo "  Background: #0a1218"
echo "  Foreground: #0c1b21"
echo ""
echo "Restart ComfyUI to see the changes!"
