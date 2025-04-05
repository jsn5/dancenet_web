/**
 * Visualization helper for rendering dance poses on canvas with 2D sprites
 */
class PoseVisualizer {
    /**
     * Initialize the pose visualizer
     * @param {HTMLCanvasElement} canvas - Canvas element for drawing
     */
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.width = canvas.width;
        this.height = canvas.height;
        
        // Default settings
        this.settings = {
            keypoint: {
                radius: 5,
                color: '#e74c3c',
                highlightColor: '#f39c12'
            },
            connection: {
                lineWidth: 3,
                color: '#3498db'
            },
            sprite: {
                color: '#3498db',
                headColor: '#e67e22'
            },
            background: '#ecf0f1'
        };
        
        // Visualization mode: 'skeleton' or 'sprite'
        this._visualizationMode = 'sprite'; // Use an internal property with a getter/setter
        
        // Load sprites
        this.sprites = {
            head: null,
            torso: null,
            arm: null,
            leg: null,
            hand: null,
            foot: null
        };
        
        // Body part definitions (indices and connections)
        this.bodyParts = {
            head: { 
                keypoints: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // Face keypoints
                center: 0 // Nose as center point
            },
            torso: {
                keypoints: [11, 12, 23, 24], // Shoulders and hips
                connections: [[11, 12], [12, 24], [24, 23], [23, 11]] // Torso outline
            },
            leftArm: {
                keypoints: [11, 13, 15], // Left shoulder, elbow, wrist
                connections: [[11, 13], [13, 15]]
            },
            rightArm: {
                keypoints: [12, 14, 16], // Right shoulder, elbow, wrist
                connections: [[12, 14], [14, 16]]
            },
            leftLeg: {
                keypoints: [23, 25, 27, 29, 31], // Left hip, knee, ankle, foot
                connections: [[23, 25], [25, 27], [27, 29], [29, 31]]
            },
            rightLeg: {
                keypoints: [24, 26, 28, 30, 32], // Right hip, knee, ankle, foot
                connections: [[24, 26], [26, 28], [28, 30], [30, 32]]
            }
        };
        
        // Keypoint connections for skeleton visualization (MediaPipe format)
        this.connections = [
            // Face connections
            [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8],
            // Upper body connections
            [9, 10], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
            // Lower body connections
            [11, 23], [12, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32]
        ];
        
        // Create basic body part shapes instead of loading images
        // This allows us to visualize without external resources
        this.createBodyPartShapes();
        
        // Log initial state
        console.log("PoseVisualizer initialized with mode:", this._visualizationMode);
    }
    
    // Add getter and setter for visualizationMode to properly handle changes
    get visualizationMode() {
        return this._visualizationMode;
    }
    
    set visualizationMode(newMode) {
        console.log("Setting visualization mode to:", newMode);
        if (newMode === 'skeleton' || newMode === 'sprite') {
            this._visualizationMode = newMode;
        } else {
            console.warn("Invalid visualization mode:", newMode, "Must be 'skeleton' or 'sprite'");
        }
    }
    
    /**
     * Create simple shapes to use as body parts
     */
    createBodyPartShapes() {
        // Create off-screen canvases for each body part
        const headCanvas = document.createElement('canvas');
        headCanvas.width = 60;
        headCanvas.height = 60;
        const headCtx = headCanvas.getContext('2d');
        
        const torsoCanvas = document.createElement('canvas');
        torsoCanvas.width = 100;
        torsoCanvas.height = 130;
        const torsoCtx = torsoCanvas.getContext('2d');
        
        const armCanvas = document.createElement('canvas');
        armCanvas.width = 100;
        armCanvas.height = 30;
        const armCtx = armCanvas.getContext('2d');
        
        const legCanvas = document.createElement('canvas');
        legCanvas.width = 40;
        legCanvas.height = 120;
        const legCtx = legCanvas.getContext('2d');
        
        const handCanvas = document.createElement('canvas');
        handCanvas.width = 25;
        handCanvas.height = 25;
        const handCtx = handCanvas.getContext('2d');
        
        const footCanvas = document.createElement('canvas');
        footCanvas.width = 40;
        footCanvas.height = 20;
        const footCtx = footCanvas.getContext('2d');
        
        // Draw head (circle)
        headCtx.beginPath();
        headCtx.arc(30, 25, 25, 0, 2 * Math.PI);
        headCtx.fillStyle = this.settings.sprite.headColor;
        headCtx.fill();
        
        // Draw torso (rounded rectangle)
        roundedRect(torsoCtx, 30, 5, 40, 120, 20);
        torsoCtx.fillStyle = this.settings.sprite.color;
        torsoCtx.fill();
        
        // Draw arm (rounded rectangle)
        roundedRect(armCtx, 5, 5, 90, 20, 10);
        armCtx.fillStyle = this.settings.sprite.color;
        armCtx.fill();
        
        // Draw leg (rounded rectangle)
        roundedRect(legCtx, 5, 40, 30, 40, 10);
        legCtx.fillStyle = this.settings.sprite.color;
        legCtx.fill();
        
        // Draw hand (circle)
        handCtx.beginPath();
        handCtx.arc(12.5, 12.5, 10, 0, 2 * Math.PI);
        handCtx.fillStyle = this.settings.sprite.headColor;
        handCtx.fill();
        
        // Draw foot (oval)
        footCtx.beginPath();
        footCtx.ellipse(20, 10, 18, 8, 0, 0, 2 * Math.PI);
        footCtx.fillStyle = this.settings.sprite.headColor;
        footCtx.fill();
        
        // Store the canvases as sprites
        this.sprites.head = headCanvas;
        this.sprites.torso = torsoCanvas;
        this.sprites.arm = armCanvas;
        this.sprites.leg = legCanvas;
        this.sprites.hand = handCanvas;
        this.sprites.foot = footCanvas;
        
        // Helper function for drawing rounded rectangles
        function roundedRect(ctx, x, y, width, height, radius) {
            ctx.beginPath();
            ctx.moveTo(x + radius, y);
            ctx.lineTo(x + width - radius, y);
            ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
            ctx.lineTo(x + width, y + height - radius);
            ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
            ctx.lineTo(x + radius, y + height);
            ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
            ctx.lineTo(x, y + radius);
            ctx.quadraticCurveTo(x, y, x + radius, y);
            ctx.closePath();
        }
    }
    
    /**
     * Clear the canvas
     */
    clear() {
        this.ctx.fillStyle = this.settings.background;
        this.ctx.fillRect(0, 0, this.width, this.height);
    }
    
    /**
     * Draw a keypoint
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} size - Size of the keypoint
     * @param {string} color - Color of the keypoint
     */
    drawKeypoint(x, y, size = this.settings.keypoint.radius, color = this.settings.keypoint.color) {
        this.ctx.beginPath();
        this.ctx.arc(x, y, size, 0, 2 * Math.PI);
        this.ctx.fillStyle = color;
        this.ctx.fill();
    }
    
    /**
     * Draw a connection between two keypoints
     * @param {number} x1 - X coordinate of the first keypoint
     * @param {number} y1 - Y coordinate of the first keypoint
     * @param {number} x2 - X coordinate of the second keypoint
     * @param {number} y2 - Y coordinate of the second keypoint
     * @param {number} lineWidth - Width of the line
     * @param {string} color - Color of the line
     */
    drawConnection(x1, y1, x2, y2, lineWidth = this.settings.connection.lineWidth, color = this.settings.connection.color) {
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.lineWidth = lineWidth;
        this.ctx.strokeStyle = color;
        this.ctx.stroke();
    }
    
    /**
     * Draw a sprite-based visualization
     * @param {Array} keypoints - Normalized keypoint coordinates
     */
    drawSprites(keypoints) {
        // Draw torso
        this.drawTorso(keypoints);
        
        // Draw arms
        this.drawLimb(keypoints, this.bodyParts.leftArm, this.sprites.arm, true);
        this.drawLimb(keypoints, this.bodyParts.rightArm, this.sprites.arm, false);
        
        // Draw legs
        this.drawLimb(keypoints, this.bodyParts.leftLeg, this.sprites.leg, true);
        this.drawLimb(keypoints, this.bodyParts.rightLeg, this.sprites.leg, false);
        
        // Draw hands
        if (keypoints[15] && keypoints[15][0] !== 0) {
            this.drawSprite(this.sprites.hand, keypoints[15][0], keypoints[15][1], 0, 0.9);
        }
        if (keypoints[16] && keypoints[16][0] !== 0) {
            this.drawSprite(this.sprites.hand, keypoints[16][0], keypoints[16][1], 0, 0.9);
        }
        
        // Draw feet
        if (keypoints[31] && keypoints[31][0] !== 0) {
            this.drawSprite(this.sprites.foot, keypoints[31][0], keypoints[31][1], 0, 1);
        }
        if (keypoints[32] && keypoints[32][0] !== 0) {
            this.drawSprite(this.sprites.foot, keypoints[32][0], keypoints[32][1], 0, 1);
        }
        
        // Draw head
        if (keypoints[0] && keypoints[0][0] !== 0) {
            // Calculate head size based on eye distance if available
            let headScale = 1.2;
            if (keypoints[2] && keypoints[5] && 
                keypoints[2][0] !== 0 && keypoints[5][0] !== 0) {
                const eyeDistance = Math.sqrt(
                    Math.pow(keypoints[2][0] - keypoints[5][0], 2) + 
                    Math.pow(keypoints[2][1] - keypoints[5][1], 2)
                );
                headScale = eyeDistance / 30;
            }
            
            this.drawSprite(this.sprites.head, keypoints[0][0], keypoints[0][1], 0, headScale);
        }
    }
    
    /**
     * Draw the torso using a sprite
     * @param {Array} keypoints - Normalized keypoint coordinates
     */
    drawTorso(keypoints) {
        // Need all four corners of the torso
        const leftShoulder = keypoints[11];
        const rightShoulder = keypoints[12];
        const leftHip = keypoints[23];
        const rightHip = keypoints[24];
        
        if (!leftShoulder || !rightShoulder || !leftHip || !rightHip ||
            leftShoulder[0] === 0 || rightShoulder[0] === 0 || 
            leftHip[0] === 0 || rightHip[0] === 0) {
            return; // Missing essential keypoints
        }
        
        // Calculate torso dimensions and rotation
        const torsoWidth = Math.sqrt(
            Math.pow(rightShoulder[0] - leftShoulder[0], 2) + 
            Math.pow(rightShoulder[1] - leftShoulder[1], 2)
        );
        
        const torsoHeight = Math.sqrt(
            Math.pow(leftShoulder[0] - leftHip[0], 2) + 
            Math.pow(leftShoulder[1] - leftHip[1], 2)
        );
        
        // Calculate center of torso
        const centerX = (leftShoulder[0] + rightShoulder[0] + leftHip[0] + rightHip[0]) / 4;
        const centerY = (leftShoulder[1] + rightShoulder[1] + leftHip[1] + rightHip[1]) / 4;
        
        // Calculate torso rotation (from shoulders)
        const angle = Math.atan2(
            rightShoulder[1] - leftShoulder[1],
            rightShoulder[0] - leftShoulder[0]
        );
        
        // Calculate scale factors
        const scaleX = torsoWidth / 60; // Adjusted for sprite width
        const scaleY = torsoHeight / 100; // Adjusted for sprite height
        
        // Draw the torso sprite with rotation and scaling
        this.ctx.save();
        this.ctx.translate(centerX, centerY);
        this.ctx.rotate(angle);
        this.ctx.drawImage(
            this.sprites.torso, 
            -this.sprites.torso.width * scaleX / 2, 
            -this.sprites.torso.height * scaleY / 2,
            this.sprites.torso.width * scaleX,
            this.sprites.torso.height * scaleY
        );
        this.ctx.restore();
    }
    
    /**
     * Draw a limb (arm or leg) using a sprite
     * @param {Array} keypoints - Normalized keypoint coordinates
     * @param {Object} limb - Limb definition (keypoints and connections)
     * @param {HTMLCanvasElement} sprite - Sprite to use for the limb
     * @param {boolean} isLeft - Whether this is a left limb
     */
    drawLimb(keypoints, limb, sprite, isLeft) {
        const kps = limb.keypoints;
        
        // Check if we have valid keypoints for this limb
        if (kps.some(idx => !keypoints[idx] || keypoints[idx][0] === 0)) {
            return; // Missing essential keypoints
        }
        
        // For arms and legs, we draw segments separately to handle rotation
        for (let i = 0; i < kps.length - 1; i++) {
            const start = keypoints[kps[i]];
            const end = keypoints[kps[i + 1]];
            
            // Calculate segment properties
            const segmentLength = Math.sqrt(
                Math.pow(end[0] - start[0], 2) + 
                Math.pow(end[1] - start[1], 2)
            );
            
            const angle = Math.atan2(
                end[1] - start[1],
                end[0] - start[0]
            );
            
            // Calculate midpoint for rotation anchor
            const midX = (start[0] + end[0]) / 2;
            const midY = (start[1] + end[1]) / 2;
            
            // Scale based on segment type (thicker for upper limbs)
            const scaleY = (i === 0) ? 0.9 : 0.7;
            const scaleX = segmentLength / sprite.width;
            
            // Draw the limb segment with rotation
            this.ctx.save();
            this.ctx.translate(midX, midY);
            this.ctx.rotate(angle);
            this.ctx.drawImage(
                sprite,
                -sprite.width * scaleX / 2,
                -sprite.height * scaleY / 2,
                sprite.width * scaleX,
                sprite.height * scaleY
            );
            this.ctx.restore();
        }
    }
    
    /**
     * Draw a sprite at a specific position with rotation and scaling
     * @param {HTMLCanvasElement} sprite - The sprite to draw
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {number} angle - Rotation angle in radians
     * @param {number} scale - Scale factor
     */
    drawSprite(sprite, x, y, angle, scale) {
        const width = sprite.width * scale;
        const height = sprite.height * scale;
        
        this.ctx.save();
        this.ctx.translate(x, y);
        this.ctx.rotate(angle);
        this.ctx.drawImage(sprite, -width/2, -height/2, width, height);
        this.ctx.restore();
    }
    
    /**
     * Draw a complete pose
     * @param {Array} keypoints - Array of keypoint coordinates [[x1, y1], [x2, y2], ...]
     * @param {boolean} normalize - Whether to normalize the coordinates to fit the canvas
     */
    drawPose(keypoints, normalize = true) {
        if (!keypoints || keypoints.length === 0) return;
        
        // Clear canvas
        this.clear();
        
        // Normalize coordinates if needed
        let normalizedKeypoints = keypoints;
        if (normalize) {
            normalizedKeypoints = this.normalizeCoordinates(keypoints);
        }
        
        // Log the visualization mode that's being used
        console.log("Drawing with mode:", this._visualizationMode);
        
        // Draw based on visualization mode
        if (this._visualizationMode === 'skeleton') {
            // Draw connections first
            for (const [i, j] of this.connections) {
                if (i >= normalizedKeypoints.length || j >= normalizedKeypoints.length) continue;
                
                const [x1, y1] = normalizedKeypoints[i];
                const [x2, y2] = normalizedKeypoints[j];
                
                if (x1 === 0 && y1 === 0) continue;
                if (x2 === 0 && y2 === 0) continue;
                
                this.drawConnection(x1, y1, x2, y2);
            }
            
            // Draw keypoints
            for (let i = 0; i < normalizedKeypoints.length; i++) {
                const [x, y] = normalizedKeypoints[i];
                if (x === 0 && y === 0) continue;
                
                this.drawKeypoint(x, y);
            }
        } else if (this._visualizationMode === 'sprite') {
            // Draw sprite-based visualization
            this.drawSprites(normalizedKeypoints);
        }
    }
    
    /**
     * Normalize keypoint coordinates to fit the canvas
     * @param {Array} keypoints - Array of keypoint coordinates [[x1, y1], [x2, y2], ...]
     * @returns {Array} - Normalized keypoints
     */
    normalizeCoordinates(keypoints) {
        if (!keypoints || keypoints.length === 0) return [];
        
        // Find the bounding box
        let minX = Infinity;
        let minY = Infinity;
        let maxX = -Infinity;
        let maxY = -Infinity;
        
        for (const [x, y] of keypoints) {
            // Skip [0, 0] coordinates (usually indicates missing keypoints)
            if (x === 0 && y === 0) continue;
            
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }
        
        // If no valid keypoints found, return the original
        if (minX === Infinity || minY === Infinity || maxX === -Infinity || maxY === -Infinity) {
            return keypoints;
        }
        
        // Calculate scale and translation
        const boxWidth = maxX - minX;
        const boxHeight = maxY - minY;
        
        const scaleX = (this.width * 0.8) / boxWidth;
        const scaleY = (this.height * 0.8) / boxHeight;
        const scale = Math.min(scaleX, scaleY);
        
        const translateX = (this.width - boxWidth * scale) / 2 - minX * scale;
        const translateY = (this.height - boxHeight * scale) / 2 - minY * scale;
        
        // Normalize keypoints
        const normalizedKeypoints = keypoints.map(([x, y]) => {
            // Skip [0, 0] coordinates
            if (x === 0 && y === 0) return [0, 0];
            
            const normalizedX = x * scale + translateX;
            const normalizedY = y * scale + translateY;
            
            return [normalizedX, normalizedY];
        });
        
        return normalizedKeypoints;
    }
    
    /**
     * Resize the canvas to maintain the aspect ratio
     */
    resize() {
        const container = this.canvas.parentElement;
        const containerWidth = container.clientWidth;
        
        // Maintain aspect ratio (original is 4:3)
        const aspectRatio = 3 / 4;
        const newHeight = containerWidth * aspectRatio;
        
        this.canvas.width = containerWidth;
        this.canvas.height = newHeight;
        
        this.width = containerWidth;
        this.height = newHeight;
    }
}
