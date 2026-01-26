import { app } from "/scripts/app.js";

/**
 * String List Cowboy - Dynamic Input Extension
 *
 * Automatically adds new value inputs when you connect to the last available one.
 * Removes empty trailing inputs when disconnected (keeping one empty slot).
 */
app.registerExtension({
    name: "Trent.StringListCowboy",

    async nodeCreated(node) {
        if (node.constructor.comfyClass !== "StringListCowboy") {
            return;
        }

        /**
         * Get all value input indices currently on the node
         */
        const getValueInputIndices = () => {
            const indices = [];
            for (const input of node.inputs || []) {
                const match = input.name.match(/^value_(\d+)$/);
                if (match) {
                    indices.push(parseInt(match[1]));
                }
            }
            return indices.sort((a, b) => a - b);
        };

        /**
         * Check if an input at given index is connected
         */
        const isInputConnected = (index) => {
            const input = node.inputs?.find(i => i.name === `value_${index}`);
            return input && input.link !== null;
        };

        /**
         * Add a new value input slot
         */
        const addValueInput = (index) => {
            const inputName = `value_${index}`;

            if (node.inputs?.find(i => i.name === inputName)) {
                return false;
            }

            node.addInput(inputName, "*");
            return true;
        };

        /**
         * Remove a value input slot
         */
        const removeValueInput = (index) => {
            const inputName = `value_${index}`;

            const inputIdx = node.inputs?.findIndex(i => i.name === inputName);
            if (inputIdx >= 0) {
                const input = node.inputs[inputIdx];
                if (input.link !== null) {
                    app.graph.removeLink(input.link);
                }
                node.removeInput(inputIdx);
            }
        };

        /**
         * Main function to update dynamic inputs based on connection state
         */
        const updateDynamicInputs = () => {
            const indices = getValueInputIndices();

            if (indices.length === 0) {
                addValueInput(1);
                return;
            }

            const connectedIndices = indices.filter(i => isInputConnected(i));
            const unconnectedIndices = indices.filter(i => !isInputConnected(i));

            const maxIndex = Math.max(...indices);

            // If the highest input is connected, add a new one
            if (isInputConnected(maxIndex)) {
                addValueInput(maxIndex + 1);
            }

            // Remove extra unconnected inputs (keep only one empty slot at the end)
            const maxConnectedIndex = connectedIndices.length > 0
                ? Math.max(...connectedIndices)
                : 0;
            const sortedUnconnected = [...unconnectedIndices].sort((a, b) => b - a);

            for (let i = 1; i < sortedUnconnected.length; i++) {
                const idx = sortedUnconnected[i];
                if (idx > maxConnectedIndex) {
                    removeValueInput(idx);
                }
            }

            // Resize node
            node.setSize(node.computeSize());
        };

        // Hook into connection changes
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, slotIndex, isConnected, link, ioSlot) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }

            // Only handle input connections (type 1)
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

            // Rebuild dynamic inputs from saved config
            if (config.inputs) {
                for (const input of config.inputs) {
                    const match = input.name.match(/^value_(\d+)$/);
                    if (match) {
                        const idx = parseInt(match[1]);
                        addValueInput(idx);
                    }
                }
            }

            setTimeout(updateDynamicInputs, 100);
        };

        // Initial setup
        setTimeout(() => {
            const indices = getValueInputIndices();
            if (indices.length === 0) {
                addValueInput(1);
            }
            updateDynamicInputs();
        }, 100);
    },
});
