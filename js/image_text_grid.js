import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

/**
 * Image+Text Grid - Dynamic Input Extension
 *
 * Automatically adds new image/caption input pairs when you connect
 * to the last available image slot. Captions can be typed directly
 * or connected from other nodes.
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
         * Check if a caption input at given index is connected
         */
        const isCaptionConnected = (index) => {
            const input = node.inputs?.find(i => i.name === `caption_${index}`);
            return input && input.link !== null;
        };

        /**
         * Find or create a caption widget for given index
         */
        const ensureCaptionWidget = (index) => {
            const captionName = `caption_${index}`;
            let widget = node.widgets?.find(w => w.name === captionName);

            if (!widget) {
                // Use ComfyWidgets.STRING for proper multiline text widget
                const result = ComfyWidgets["STRING"](
                    node,
                    captionName,
                    ["STRING", { multiline: true }],
                    app
                );
                widget = result.widget;
                widget.value = "";
            }

            return widget;
        };

        /**
         * Add a new image input slot
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
         * Add a new caption input slot
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
         * Remove an image input slot and its caption widget/input
         */
        const removeImageInput = (index) => {
            const inputName = `image_${index}`;
            const captionName = `caption_${index}`;

            // Remove image input
            const inputIdx = node.inputs?.findIndex(i => i.name === inputName);
            if (inputIdx >= 0) {
                const input = node.inputs[inputIdx];
                if (input.link !== null) {
                    app.graph.removeLink(input.link);
                }
                node.removeInput(inputIdx);
            }

            // Remove caption input
            const captionInputIdx = node.inputs?.findIndex(
                i => i.name === captionName
            );
            if (captionInputIdx >= 0) {
                const input = node.inputs[captionInputIdx];
                if (input.link !== null) {
                    app.graph.removeLink(input.link);
                }
                node.removeInput(captionInputIdx);
            }

            // Remove caption widget if exists
            const widgetIdx = node.widgets?.findIndex(w => w.name === captionName);
            if (widgetIdx >= 0) {
                node.widgets.splice(widgetIdx, 1);
            }
        };

        /**
         * Sort caption widgets so they appear in numerical order
         */
        const sortCaptionWidgets = () => {
            if (!node.widgets) return;

            const captionWidgets = [];
            const otherWidgets = [];

            for (const widget of node.widgets) {
                if (widget.name.match(/^caption_\d+$/)) {
                    captionWidgets.push(widget);
                } else {
                    otherWidgets.push(widget);
                }
            }

            captionWidgets.sort((a, b) => {
                const aNum = parseInt(a.name.match(/^caption_(\d+)$/)[1]);
                const bNum = parseInt(b.name.match(/^caption_(\d+)$/)[1]);
                return aNum - bNum;
            });

            node.widgets.length = 0;
            node.widgets.push(...otherWidgets, ...captionWidgets);
        };

        /**
         * Update widget visibility - show caption widget only when:
         * - Image is connected AND caption input is NOT connected
         */
        const updateWidgetVisibility = () => {
            const indices = getImageInputIndices();

            for (const idx of indices) {
                const captionName = `caption_${idx}`;
                const widget = node.widgets?.find(w => w.name === captionName);
                const imageConnected = isInputConnected(idx);
                const captionConnected = isCaptionConnected(idx);

                if (imageConnected && !captionConnected) {
                    // Image connected, caption not connected - show widget
                    if (!widget) {
                        ensureCaptionWidget(idx);
                    }
                } else {
                    // Either image not connected or caption is connected
                    // - remove widget
                    if (widget) {
                        const widgetIdx = node.widgets.indexOf(widget);
                        if (widgetIdx >= 0) {
                            node.widgets.splice(widgetIdx, 1);
                        }
                    }
                }
            }

            sortCaptionWidgets();
        };

        /**
         * Main function to update dynamic inputs based on connection state
         */
        const updateDynamicInputs = () => {
            const indices = getImageInputIndices();

            if (indices.length === 0) {
                addImageInput(1);
                addCaptionInput(1);
                return;
            }

            const connectedIndices = indices.filter(i => isInputConnected(i));
            const unconnectedIndices = indices.filter(i => !isInputConnected(i));

            const maxIndex = Math.max(...indices);

            // If the highest input is connected, add a new pair
            if (isInputConnected(maxIndex)) {
                addImageInput(maxIndex + 1);
                addCaptionInput(maxIndex + 1);
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

            // Remove extra unconnected inputs (keep only one empty slot)
            const maxConnectedIndex = connectedIndices.length > 0
                ? Math.max(...connectedIndices)
                : 0;
            const sortedUnconnected = [...unconnectedIndices].sort(
                (a, b) => b - a
            );

            for (let i = 1; i < sortedUnconnected.length; i++) {
                const idx = sortedUnconnected[i];
                if (idx > maxConnectedIndex) {
                    removeImageInput(idx);
                }
            }

            updateWidgetVisibility();
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

        // Initial setup - ensure image_1 and caption_1 exist
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
