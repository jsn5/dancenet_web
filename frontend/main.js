/**
 * Main script for the Dance Generator frontend
 */
document.addEventListener('DOMContentLoaded', async () => {
    // Elements
    const canvas = document.getElementById('dance-canvas');
    const generateBtn = document.getElementById('generate-btn');
    const stopBtn = document.getElementById('stop-btn');
    const vizModeBtn = document.getElementById('viz-mode-btn');
    const vizModeLabel = document.getElementById('viz-mode-label');
    const temperatureSlider = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const loadingContainer = document.querySelector('.loading-container');
    const loadingText = document.getElementById('loading-text');
    
    // State
    let session = null;
    let isGenerating = false;
    let animationFrameId = null;
    let currentSequence = null;
    let seedSequence = null;
    let poseHistory = []; // Store recent poses for smoothing
    const MAX_HISTORY = 3; // Number of poses to keep for smoothing
    
    // Visualization modes - use only the modes available in visualize.js
    const visualizationModes = ['skeleton', 'sprite'];
    let currentModeIndex = 1; // Start with sprite mode (index 1)
    
    // Initialize pose visualizer
    const visualizer = new PoseVisualizer(canvas);
    visualizer.resize();
    
    // Make sure visualizer is set to sprite mode initially
    visualizer.visualizationMode = 'sprite';
    vizModeLabel.textContent = 'Mode: Sprite';
    
    // Handle window resize
    window.addEventListener('resize', () => {
        visualizer.resize();
        if (currentSequence) {
            visualizer.drawPose(currentSequence);
        }
    });
    
    // Update temperature value display
    temperatureSlider.addEventListener('input', () => {
        temperatureValue.textContent = temperatureSlider.value;
    });
    
    // Toggle visualization mode button
    vizModeBtn.addEventListener('click', () => {
        // Simple toggle between the two available modes
        currentModeIndex = (currentModeIndex + 1) % visualizationModes.length;
        const newMode = visualizationModes[currentModeIndex];
        
        // Set the mode and update the display
        visualizer.visualizationMode = newMode;
        vizModeLabel.textContent = `Mode: ${newMode.charAt(0).toUpperCase() + newMode.slice(1)}`;
        
        console.log("Changed visualization mode to:", newMode); // Debug log
        
        // Redraw current pose with new visualization mode
        if (currentSequence) {
            visualizer.drawPose(currentSequence);
        }
    });
    
    // Generate button
    generateBtn.addEventListener('click', async () => {
        if (!session) {
            showError('Model not loaded. Please wait or refresh the page.');
            return;
        }
        
        if (isGenerating) {
            return;
        }
        
        startGeneration();
    });
    
    // Stop button
    stopBtn.addEventListener('click', () => {
        stopGeneration();
    });
    
    // Initialize the app
    async function initialize() {
        try {
            // Load the ONNX model
            await loadModel();
            
            // Load or create seed sequence
            await loadSeedSequence();
            
            // Hide loading container
            loadingContainer.classList.add('hidden');
            
            // Enable generate button
            generateBtn.disabled = false;
            
            // Draw initial pose
            if (seedSequence && seedSequence.length > 0) {
                currentSequence = reshapeKeypoints(seedSequence[seedSequence.length - 1]);
                visualizer.drawPose(currentSequence);
            }
        } catch (error) {
            console.error('Initialization error:', error);
            loadingText.textContent = `Error: ${error.message}`;
            loadingText.style.color = 'red';
        }
    }
    
    /**
     * Load the ONNX model
     */
    async function loadModel() {
        try {
            loadingText.textContent = 'Loading ONNX model...';
            
            // Set WebAssembly execution provider options
            const options = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            };
            
            // Create session
            session = await ort.InferenceSession.create('./model.onnx', options);
            
            loadingText.textContent = 'Model loaded successfully!';
        } catch (error) {
            console.error('Error loading model:', error);
            throw new Error('Failed to load model. Check console for details.');
        }
    }
    
    /**
     * Load or create seed sequence
     */
    async function loadSeedSequence() {
        try {
            loadingText.textContent = 'Loading seed sequence...';
            
            // Try to load a pre-defined seed sequence
            const response = await fetch('./seed_sequence.json');
            
            if (response.ok) {
                const data = await response.json();
                seedSequence = data.sequence;
                loadingText.textContent = 'Seed sequence loaded!';
            } else {
                // If no seed sequence is available, use a default one
                // This is a placeholder - in a real app, you'd need actual pose data
                console.warn('No seed sequence found, using default');
                
                // Create a placeholder sequence with the right dimensions
                // Assuming 33 keypoints (MediaPipe format) flattened to 66 values
                const seqLength = 30;
                const numKeypoints = 33;
                seedSequence = Array(seqLength).fill().map(() => Array(numKeypoints * 2).fill(0));
                
                // Set some basic keypoints for visualization
                // This creates a simple stick figure
                for (let i = 0; i < seqLength; i++) {
                    // Head at the center top
                    seedSequence[i][0] = 400; // nose x
                    seedSequence[i][1] = 100; // nose y
                    
                    // Shoulders
                    seedSequence[i][22] = 350; // left shoulder x
                    seedSequence[i][23] = 150; // left shoulder y
                    seedSequence[i][24] = 450; // right shoulder x
                    seedSequence[i][25] = 150; // right shoulder y
                    
                    // Hips
                    seedSequence[i][46] = 350; // left hip x
                    seedSequence[i][47] = 300; // left hip y
                    seedSequence[i][48] = 450; // right hip x
                    seedSequence[i][49] = 300; // right hip y
                    
                    // Knees
                    seedSequence[i][50] = 350; // left knee x
                    seedSequence[i][51] = 400; // left knee y
                    seedSequence[i][52] = 450; // right knee x
                    seedSequence[i][53] = 400; // right knee y
                    
                    // Ankles
                    seedSequence[i][54] = 350; // left ankle x
                    seedSequence[i][55] = 500; // left ankle y
                    seedSequence[i][56] = 450; // right ankle x
                    seedSequence[i][57] = 500; // right ankle y
                }
                
                loadingText.textContent = 'Using default seed sequence';
            }
        } catch (error) {
            console.error('Error loading seed sequence:', error);
            throw new Error('Failed to load seed sequence');
        }
    }
    
    /**
     * Start generating dance poses
     */
    function startGeneration() {
        if (!session || !seedSequence || isGenerating) return;
        
        isGenerating = true;
        generateBtn.disabled = true;
        stopBtn.disabled = false;
        
        // Reset pose history
        poseHistory = [];
        
        // Use the seed sequence as the current input
        let inputSequence = [...seedSequence];
        
        // Start the generation loop
        runGenerationStep(inputSequence);
    }
    
    /**
     * Apply a simple smoothing to keypoints using recent history
     * @param {Array} newPose - New pose to smooth
     * @returns {Array} - Smoothed pose
     */
    function smoothPose(newPose) {
        // If no history or smoothing is minimal, return original pose
        if (poseHistory.length === 0) {
            return newPose;
        }
        
        // Create a smoothed pose with same structure as new pose
        const smoothedPose = [];
        
        // For each keypoint in the pose
        for (let i = 0; i < newPose.length; i++) {
            // Skip keypoints that are [0,0] in the new pose
            if (newPose[i][0] === 0 && newPose[i][1] === 0) {
                smoothedPose.push([0, 0]);
                continue;
            }
            
            // Start with the new pose values
            let smoothedX = newPose[i][0];
            let smoothedY = newPose[i][1];
            let validHistoryCount = 0;
            
            // Apply decreasing weight to historical poses
            for (let j = 0; j < poseHistory.length; j++) {
                const historicalPose = poseHistory[j];
                
                // Skip if this keypoint is missing in the historical pose
                if (i >= historicalPose.length || 
                    (historicalPose[i][0] === 0 && historicalPose[i][1] === 0)) {
                    continue;
                }
                
                // Weight based on recency (more recent = higher weight)
                // This creates a simple exponential moving average
                const weight = 0.6 * Math.pow(0.5, j);
                
                // Apply weighted average
                smoothedX = smoothedX * (1 - weight) + historicalPose[i][0] * weight;
                smoothedY = smoothedY * (1 - weight) + historicalPose[i][1] * weight;
                validHistoryCount++;
            }
            
            smoothedPose.push([smoothedX, smoothedY]);
        }
        
        return smoothedPose;
    }
    
    /**
     * Run a single generation step
     * @param {Array} inputSequence - The input sequence for the model
     */
    async function runGenerationStep(inputSequence) {
        if (!isGenerating) return;
        
        try {
            // Create input tensor
            const inputTensor = new ort.Tensor(
                'float32',
                new Float32Array(inputSequence.flat()),
                [1, inputSequence.length, inputSequence[0].length]
            );
            
            // Run inference
            const feeds = { input_sequence: inputTensor };
            const results = await session.run(feeds);
            
            // Extract the outputs
            const pi = results.pi.data;
            const mu = results.mu.data;
            const sigma = results.sigma.data;
            
            // Sample from the mixture model
            const temperature = parseFloat(temperatureSlider.value);
            const nextPose = sampleFromMixture(
                pi, 
                mu, 
                sigma, 
                inputSequence[0].length, 
                results.mu.dims[1], 
                temperature
            );
            
            // Update the sequence by removing the first pose and adding the new one
            inputSequence.shift();
            inputSequence.push(nextPose);
            
            // Reshape the generated pose
            const newPose = reshapeKeypoints(nextPose);
            
            // Add the new pose to the history
            poseHistory.unshift(newPose);
            if (poseHistory.length > MAX_HISTORY) {
                poseHistory.pop();
            }
            
            // Apply smoothing to the new pose
            const smoothedPose = smoothPose(newPose);
            
            // Update the current pose for visualization
            currentSequence = smoothedPose;
            visualizer.drawPose(currentSequence);
            
            // Add a small delay to reduce generation frequency (makes animation smoother)
            setTimeout(() => {
                // Schedule the next generation step
                animationFrameId = requestAnimationFrame(() => runGenerationStep(inputSequence));
            }, 30); // 30ms delay helps reduce jitter while maintaining responsiveness
        } catch (error) {
            console.error('Generation error:', error);
            stopGeneration();
            showError('Error during generation. Check console for details.');
        }
    }
    
    /**
     * Stop the generation process
     */
    function stopGeneration() {
        isGenerating = false;
        generateBtn.disabled = false;
        stopBtn.disabled = true;
        
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    }
    
    /**
     * Sample from a Gaussian mixture model
     * @param {Float32Array} pi - Mixture weights
     * @param {Float32Array} mu - Means
     * @param {Float32Array} sigma - Standard deviations
     * @param {number} outputSize - Size of the output vector
     * @param {number} numMixtures - Number of mixture components
     * @param {number} temperature - Temperature for sampling
     * @returns {Array} - Sampled pose vector
     */
    function sampleFromMixture(pi, mu, sigma, outputSize, numMixtures, temperature = 1.0) {
        // Apply temperature to mixture weights
        let weights = new Array(numMixtures);
        
        if (temperature !== 1.0) {
            // Scale logits by temperature
            const scaledPi = new Array(numMixtures);
            for (let i = 0; i < numMixtures; i++) {
                scaledPi[i] = pi[i] / temperature;
            }
            
            // Apply softmax
            const maxLogit = Math.max(...scaledPi);
            let sumExp = 0;
            
            for (let i = 0; i < numMixtures; i++) {
                weights[i] = Math.exp(scaledPi[i] - maxLogit);
                sumExp += weights[i];
            }
            
            for (let i = 0; i < numMixtures; i++) {
                weights[i] /= sumExp;
            }
        } else {
            // Use the original pi values
            for (let i = 0; i < numMixtures; i++) {
                weights[i] = pi[i];
            }
        }
        
        // Sample component index based on weights
        const componentIdx = sampleFromCategorical(weights);
        
        // Get means and standard deviations for the selected component
        const selectedMu = mu.slice(
            componentIdx * outputSize, 
            (componentIdx + 1) * outputSize
        );
        
        const selectedSigma = sigma.slice(
            componentIdx * outputSize, 
            (componentIdx + 1) * outputSize
        );
        
        // Scale sigma by temperature (higher temperature = more variance)
        const scaledSigma = selectedSigma.map(s => s * temperature);
        
        // Sample from the Gaussian distribution
        const sample = new Array(outputSize);
        for (let i = 0; i < outputSize; i++) {
            sample[i] = sampleFromGaussian(selectedMu[i], scaledSigma[i]);
        }
        
        return sample;
    }
    
    /**
     * Sample from a categorical distribution
     * @param {Array} probs - Probability distribution
     * @returns {number} - Sampled index
     */
    function sampleFromCategorical(probs) {
        const r = Math.random();
        let cumSum = 0;
        
        for (let i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (r < cumSum) {
                return i;
            }
        }
        
        // Fallback to the most probable component
        return probs.indexOf(Math.max(...probs));
    }
    
    /**
     * Sample from a Gaussian distribution
     * @param {number} mean - Mean of the distribution
     * @param {number} stddev - Standard deviation
     * @returns {number} - Sampled value
     */
    function sampleFromGaussian(mean, stddev) {
        // Box-Muller transform for Gaussian sampling
        const u1 = Math.random();
        const u2 = Math.random();
        
        const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        
        return mean + stddev * z0;
    }
    
    /**
     * Reshape a flat array of keypoints to a 2D array
     * @param {Array} flatKeypoints - Flat array of keypoints [x1, y1, x2, y2, ...]
     * @returns {Array} - 2D array of keypoints [[x1, y1], [x2, y2], ...]
     */
    function reshapeKeypoints(flatKeypoints) {
        const keypoints = [];
        
        for (let i = 0; i < flatKeypoints.length; i += 2) {
            keypoints.push([flatKeypoints[i], flatKeypoints[i + 1]]);
        }
        
        return keypoints;
    }
    
    /**
     * Show an error message to the user
     * @param {string} message - Error message
     */
    function showError(message) {
        // Replace with your preferred UI for showing errors
        alert(message);
    }
    
    // Start initialization
    initialize();
});
