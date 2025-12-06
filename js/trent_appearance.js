import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "TrentNodes.appearance",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Check if this is a Trent node by looking at the category
        if (nodeData.category && nodeData.category.startsWith("Trent")) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Apply Trent Nodes theme
                this.bgcolor = "#0a1218";  // Dark background
                this.color = "#0c1b21";    // Darker teal header
                
                return r;
            };
        }
    }
});
