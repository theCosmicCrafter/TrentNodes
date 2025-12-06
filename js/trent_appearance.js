import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "TrentNodes.appearance",
    async nodeCreated(node) {
        // Apply custom colors to all nodes in Trent categories
        if (node.comfyClass && (
            node.comfyClass.includes("EnhancedVideoCutter") ||
            node.comfyClass.includes("UltimateSceneDetect") ||
            node.comfyClass.includes("VideoFolderAnalyzer") ||
            node.comfyClass.includes("LatestVideoLastNFrames") ||
            node.comfyClass.includes("LatestVideoFinalFrame") ||
            node.comfyClass.includes("CrossDissolveWithOverlap") ||
            node.comfyClass.includes("ImprovedAnimationProcessor") ||
            node.comfyClass.includes("BevelEmboss") ||
            node.comfyClass.includes("ImageBatchAnalyzer") ||
            node.comfyClass.includes("SmartFileTransfer") ||
            node.comfyClass.includes("CustomFilenameGenerator") ||
            node.comfyClass.includes("FilenameExtractor") ||
            node.comfyClass.includes("JSONSummary") ||
            node.comfyClass.includes("JSONExtractor") ||
            node.comfyClass.includes("NumberCounter") ||
            node.comfyClass.includes("Wan2.1FrameAdjuster") ||
            node.comfyClass.includes("LatentAlignedMask") ||
            node.comfyClass.includes("WanVaceKeyframeBuilder") ||
            node.comfyClass.includes("WanMagic")
        )) {
            // Trent Nodes custom theme
            node.bgcolor = "#0a1218";  // Dark background
            node.color = "#0c1b21";    // Darker teal header
        }
    }
});
