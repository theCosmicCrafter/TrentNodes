import { app } from "/scripts/app.js";

/**
 * Image+Text Grid - Dynamic Input Extension
 *
 * Automatically adds new image/caption input pairs when you connect
 * to the last available image slot. Captions are input-only (connectable).
 */
app.registerExtension({
    name: "Trent.ImageTextGrid",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "ImageTextGrid") {
            return;
        }

        /**
         * Get all image input indices currently on the node
         */
        const getImageInputIndices = () => {
            const indices = [];
            for (const input of node.inputs || []) {
                const match = input.name.match(/^image_(\d+)$/);
                if (match) {
                    indices.push(parseInt(match[1]));
                }
            }
            return indices.sort((a, b) => a - b);
        };

        /**
         * Check if an image input at given index is connected
         */
        const isInputConnected = (index) => {
            const input = node.inputs?.find(i => i.name === `image_${index}`);
            return input && input.link !== null;
        };

        /**
         * Add a new image input slot at specific index
         */
        const addImageInput = (index) => {
            const inputName = `image_${index}`;

            if (node.inputs?.find(i => i.name === inputName)) {
                return false;
            }

            node.addInput(inputName, "IMAGE");
            return true;
        };

        /**
         * Add a new caption input slot at specific index
         */
        const addCaptionInput = (index) => {
            const inputName = `caption_${index}`;

            if (node.inputs?.find(i => i.name === inputName)) {
                return false;
            }

            node.addInput(inputName, "STRING");
            return true;
        };

        /**
         * Remove an image input and its caption input by index
         */
        const removeInputPair = (index) => {
            const imageName = `image_${index}`;
            const captionName = `caption_${index}`;

            // Remove image input
            const imageIdx = node.inputs?.findIndex(i => i.name === imageName);
            if (imageIdx >= 0) {
                const input = node.inputs[imageIdx];
                if (input.link !== null) {
                    app.graph.removeLink(input.link);
                }
                node.removeInput(imageIdx);
            }

            // Remove caption input
            const captionIdx = node.inputs?.findIndex(i => i.name === captionName);
            if (captionIdx >= 0) {
                const input = node.inputs[captionIdx];
                if (input.link !== null) {
                    app.graph.removeLink(input.link);
                }
                node.removeInput(captionIdx);
            }
        };

        /**
         * Reorder inputs so they appear as image_1, caption_1, image_2, caption_2...
         */
        const reorderInputs = () => {
            if (!node.inputs) return;

            const imageInputs = [];
            const captionInputs = [];
            const otherInputs = [];

            for (const input of node.inputs) {
                const imageMatch = input.name.match(/^image_(\d+)$/);
                const captionMatch = input.name.match(/^caption_(\d+)$/);

                if (imageMatch) {
                    imageInputs.push({
                        input,
                        index: parseInt(imageMatch[1])
                    });
                } else if (captionMatch) {
                    captionInputs.push({
                        input,
                        index: parseInt(captionMatch[1])
                    });
                } else {
                    otherInputs.push(input);
                }
            }

            // Sort by index
            imageInputs.sort((a, b) => a.index - b.index);
            captionInputs.sort((a, b) => a.index - b.index);

            // Rebuild inputs array: other inputs first, then pairs
            node.inputs.length = 0;
            node.inputs.push(...otherInputs);

            // Interleave image and caption inputs
            const indices = [
                ...new Set([
                    ...imageInputs.map(i => i.index),
                    ...captionInputs.map(i => i.index)
                ])
            ].sort((a, b) => a - b);

            for (const idx of indices) {
                const img = imageInputs.find(i => i.index === idx);
                const cap = captionInputs.find(i => i.index === idx);
                if (img) node.inputs.push(img.input);
                if (cap) node.inputs.push(cap.input);
            }
        };

        /**
         * Main function to update dynamic inputs based on connection state
         */
        const updateDynamicInputs = () => {
            const indices = getImageInputIndices();

            if (indices.length === 0) {
                addImageInput(1);
                addCaptionInput(1);
                reorderInputs();
                node.setSize(node.computeSize());
                return;
            }

            // Ensure all image inputs have corresponding caption inputs
            for (const idx of indices) {
                const captionExists = node.inputs?.find(
                    i => i.name === `caption_${idx}`
                );
                if (!captionExists) {
                    addCaptionInput(idx);
                }
            }

            const connectedIndices = indices.filter(i => isInputConnected(i));
            const unconnectedIndices = indices.filter(i => !isInputConnected(i));
            const maxIndex = Math.max(...indices);

            // If the highest input is connected, add a new pair
            if (isInputConnected(maxIndex)) {
                addImageInput(maxIndex + 1);
                addCaptionInput(maxIndex + 1);
            }

            // Remove extra unconnected pairs (keep only one empty slot)
            const maxConnectedIndex = connectedIndices.length > 0
                ? Math.max(...connectedIndices)
                : 0;
            const sortedUnconnected = [...unconnectedIndices].sort(
                (a, b) => b - a
            );

            for (let i = 1; i < sortedUnconnected.length; i++) {
                const idx = sortedUnconnected[i];
                if (idx > maxConnectedIndex) {
                    removeInputPair(idx);
                }
            }

            reorderInputs();
            node.setSize(node.computeSize());
        };

        // Hook into connection changes
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(
            type, slotIndex, isConnected, link, ioSlot
        ) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }

            // Type 1 = input connections
            if (type === 1) {
                setTimeout(updateDynamicInputs, 50);
            }
        };

        // Hook into configure for loading saved workflows
        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function(config) {
            if (originalOnConfigure) {
                originalOnConfigure.apply(this, arguments);
            }

            if (config.inputs) {
                for (const input of config.inputs) {
                    const imageMatch = input.name.match(/^image_(\d+)$/);
                    if (imageMatch) {
                        const idx = parseInt(imageMatch[1]);
                        addImageInput(idx);
                        addCaptionInput(idx);
                    }
                }
            }

            setTimeout(() => {
                updateDynamicInputs();
            }, 100);
        };

        // Initial setup
        setTimeout(() => {
            const indices = getImageInputIndices();
            if (indices.length === 0) {
                addImageInput(1);
            }

            // Ensure caption_1 input exists
            const caption1Exists = node.inputs?.find(
                i => i.name === "caption_1"
            );
            if (!caption1Exists) {
                addCaptionInput(1);
            }

            updateDynamicInputs();
        }, 100);
    },
});
