import { app } from "/scripts/app.js";

/**
 * Image+Text Grid - Dynamic Input Extension
 *
 * Adds image/caption input pairs dynamically.
 * All inputs are connectable slots - no text widgets.
 */
app.registerExtension({
    name: "Trent.ImageTextGrid",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "ImageTextGrid") {
            return;
        }

        /**
         * Check if an input with given name exists
         */
        const hasInput = (name) => {
            return node.inputs?.some(i => i.name === name);
        };

        /**
         * Check if an input is connected
         */
        const isConnected = (name) => {
            const input = node.inputs?.find(i => i.name === name);
            return input && input.link !== null;
        };

        /**
         * Get the highest image index currently on the node
         */
        const getMaxImageIndex = () => {
            let max = 0;
            for (const input of node.inputs || []) {
                const match = input.name.match(/^image_(\d+)$/);
                if (match) {
                    max = Math.max(max, parseInt(match[1]));
                }
            }
            return max;
        };

        /**
         * Add an image/caption pair at given index
         */
        const addPair = (index) => {
            const imageName = `image_${index}`;
            const captionName = `caption_${index}`;

            if (!hasInput(imageName)) {
                node.addInput(imageName, "IMAGE");
            }
            if (!hasInput(captionName)) {
                node.addInput(captionName, "STRING");
            }
        };

        /**
         * Remove an image/caption pair at given index
         */
        const removePair = (index) => {
            const imageName = `image_${index}`;
            const captionName = `caption_${index}`;

            // Remove caption first (higher index after image)
            let captionIdx = node.inputs?.findIndex(i => i.name === captionName);
            if (captionIdx >= 0) {
                if (node.inputs[captionIdx].link !== null) {
                    app.graph.removeLink(node.inputs[captionIdx].link);
                }
                node.removeInput(captionIdx);
            }

            // Remove image
            let imageIdx = node.inputs?.findIndex(i => i.name === imageName);
            if (imageIdx >= 0) {
                if (node.inputs[imageIdx].link !== null) {
                    app.graph.removeLink(node.inputs[imageIdx].link);
                }
                node.removeInput(imageIdx);
            }
        };

        /**
         * Reorder inputs: image_1, caption_1, image_2, caption_2, ...
         */
        const reorderInputs = () => {
            if (!node.inputs || node.inputs.length === 0) return;

            // Separate dynamic inputs from others
            const pairs = new Map(); // index -> {image, caption}
            const others = [];

            for (const input of node.inputs) {
                const imageMatch = input.name.match(/^image_(\d+)$/);
                const captionMatch = input.name.match(/^caption_(\d+)$/);

                if (imageMatch) {
                    const idx = parseInt(imageMatch[1]);
                    if (!pairs.has(idx)) pairs.set(idx, {});
                    pairs.get(idx).image = input;
                } else if (captionMatch) {
                    const idx = parseInt(captionMatch[1]);
                    if (!pairs.has(idx)) pairs.set(idx, {});
                    pairs.get(idx).caption = input;
                } else {
                    others.push(input);
                }
            }

            // Sort indices
            const sortedIndices = [...pairs.keys()].sort((a, b) => a - b);

            // Rebuild inputs array
            node.inputs.length = 0;

            // Add pairs in order: image_N, caption_N
            for (const idx of sortedIndices) {
                const pair = pairs.get(idx);
                if (pair.image) node.inputs.push(pair.image);
                if (pair.caption) node.inputs.push(pair.caption);
            }

            // Add other inputs at the end
            node.inputs.push(...others);
        };

        /**
         * Update dynamic inputs based on connections
         */
        const updateInputs = () => {
            const maxIdx = getMaxImageIndex();

            // Always have at least pair 1
            if (maxIdx === 0) {
                addPair(1);
                reorderInputs();
                node.setSize(node.computeSize());
                return;
            }

            // Ensure all pairs exist
            for (let i = 1; i <= maxIdx; i++) {
                addPair(i);
            }

            // Find highest connected image
            let highestConnected = 0;
            for (let i = 1; i <= maxIdx; i++) {
                if (isConnected(`image_${i}`)) {
                    highestConnected = i;
                }
            }

            // If highest index is connected, add next pair
            if (isConnected(`image_${maxIdx}`)) {
                addPair(maxIdx + 1);
            }

            // Remove trailing empty pairs (keep one empty)
            const newMax = getMaxImageIndex();
            let emptyCount = 0;
            for (let i = newMax; i > highestConnected; i--) {
                if (!isConnected(`image_${i}`)) {
                    emptyCount++;
                    if (emptyCount > 1) {
                        removePair(i);
                    }
                }
            }

            reorderInputs();
            node.setSize(node.computeSize());
        };

        // Remove any caption widgets that might exist
        const removeWidgets = () => {
            if (!node.widgets) return;
            node.widgets = node.widgets.filter(w => {
                return !w.name.match(/^caption_\d+$/);
            });
        };

        // Hook connection changes
        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, slot, connected, link, io) {
            if (origOnConnectionsChange) {
                origOnConnectionsChange.apply(this, arguments);
            }
            if (type === 1) { // Input
                removeWidgets();
                setTimeout(updateInputs, 50);
            }
        };

        // Hook configure for loading workflows
        const origOnConfigure = node.onConfigure;
        node.onConfigure = function(config) {
            if (origOnConfigure) {
                origOnConfigure.apply(this, arguments);
            }

            // Add pairs for any saved inputs
            if (config.inputs) {
                const indices = new Set();
                for (const input of config.inputs) {
                    const match = input.name.match(/^image_(\d+)$/);
                    if (match) {
                        indices.add(parseInt(match[1]));
                    }
                }
                for (const idx of indices) {
                    addPair(idx);
                }
            }

            setTimeout(() => {
                removeWidgets();
                updateInputs();
            }, 100);
        };

        // Initial setup
        setTimeout(() => {
            removeWidgets();
            if (!hasInput("image_1")) {
                addPair(1);
            } else if (!hasInput("caption_1")) {
                // image_1 exists but caption_1 doesn't
                node.addInput("caption_1", "STRING");
            }
            updateInputs();
        }, 100);
    },
});
