// Three-Body Problem Real-Time Visualization Application
// Main JavaScript file for handling user interactions and visualization

class ThreeBodyVisualizer {
    constructor() {
        this.currentResults = null;
        this.presets = {};
        this.isRunning = false;
        
        this.initializeEventListeners();
        this.loadPresets();
        this.initializePlots();
    }
    
    initializeEventListeners() {
        // Range slider event listeners
        document.getElementById('integration-time').addEventListener('input', (e) => {
            document.getElementById('integration-time-value').textContent = e.target.value + 's';
        });
        
        document.getElementById('time-step').addEventListener('input', (e) => {
            document.getElementById('time-step-value').textContent = e.target.value + 's';
        });
        
        document.getElementById('size-factor').addEventListener('input', (e) => {
            document.getElementById('size-factor-value').textContent = e.target.value;
        });
        
        document.getElementById('velocity-factor').addEventListener('input', (e) => {
            document.getElementById('velocity-factor-value').textContent = e.target.value;
        });
        
        document.getElementById('rotation-rate').addEventListener('input', (e) => {
            document.getElementById('rotation-rate-value').textContent = e.target.value;
        });
        
        // Configuration type change
        document.getElementById('config-type').addEventListener('change', (e) => {
            this.updateConfigurationUI(e.target.value);
        });
        
        // Preset selection
        document.getElementById('preset-select').addEventListener('change', (e) => {
            if (e.target.value) {
                this.loadPreset(e.target.value);
            }
        });
        
        // Action buttons
        document.getElementById('run-simulation').addEventListener('click', () => {
            this.runSimulation();
        });
        
        document.getElementById('reset-simulation').addEventListener('click', () => {
            this.resetSimulation();
        });
        
        document.getElementById('poincare-body').addEventListener('change', () => this.updatePoincarePlot());
        document.getElementById('poincare-plane').addEventListener('change', () => this.updatePoincarePlot());
        document.getElementById('phase-body').addEventListener('change', () => this.updatePhasePlot());
        document.getElementById('phase-x').addEventListener('change', () => this.updatePhasePlot());
        document.getElementById('phase-vx').addEventListener('change', () => this.updatePhasePlot());
        document.getElementById('download-data').addEventListener('click', () => this.downloadData());
    }
    
    async loadPresets() {
        try {
            const response = await fetch('/api/presets');
            this.presets = await response.json();
            
            const presetSelect = document.getElementById('preset-select');
            presetSelect.innerHTML = '<option value="">Select a preset...</option>';
            
            Object.entries(this.presets).forEach(([key, preset]) => {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = preset.name;
                presetSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Error loading presets:', error);
        }
    }
    
    loadPreset(presetKey) {
        const preset = this.presets[presetKey];
        if (!preset) return;
        
        // Update form values
        document.getElementById('config-type').value = preset.configuration_type;
        document.getElementById('mass1').value = preset.masses[0];
        document.getElementById('mass2').value = preset.masses[1];
        document.getElementById('mass3').value = preset.masses[2];
        document.getElementById('integration-time').value = preset.integration_time;
        document.getElementById('integration-time-value').textContent = preset.integration_time + 's';
        document.getElementById('size-factor').value = preset.size_factor;
        document.getElementById('size-factor-value').textContent = preset.size_factor;
        
        if (preset.velocity_factor !== undefined) {
            document.getElementById('velocity-factor').value = preset.velocity_factor;
            document.getElementById('velocity-factor-value').textContent = preset.velocity_factor;
        }
        
        if (preset.rotation_rate !== undefined) {
            document.getElementById('rotation-rate').value = preset.rotation_rate || 1.0;
            document.getElementById('rotation-rate-value').textContent = preset.rotation_rate || 1.0;
        }
        
        // Update UI based on configuration type
        this.updateConfigurationUI(preset.configuration_type);
    }
    
    updateConfigurationUI(configType) {
        const velocityGroup = document.getElementById('velocity-factor-group');
        const rotationGroup = document.getElementById('rotation-rate-group');
        const userDrawnGroup = document.getElementById('user-drawn-group');
        
        if (configType === 'homothetic') {
            velocityGroup.style.display = 'block';
            rotationGroup.style.display = 'none';
            userDrawnGroup.style.display = 'none';
        } else if (configType === 'lagrangian') {
            velocityGroup.style.display = 'none';
            rotationGroup.style.display = 'block';
            userDrawnGroup.style.display = 'none';
        } else if (configType === 'user_drawn') {
            velocityGroup.style.display = 'none';
            rotationGroup.style.display = 'none';
            userDrawnGroup.style.display = 'block';
        } else {
            velocityGroup.style.display = 'none';
            rotationGroup.style.display = 'none';
            userDrawnGroup.style.display = 'none';
        }
    }
    
    getSimulationParameters() {
        const configType = document.getElementById('config-type').value;
        const parameters = {
            masses: [
                parseFloat(document.getElementById('mass1').value),
                parseFloat(document.getElementById('mass2').value),
                parseFloat(document.getElementById('mass3').value)
            ],
            configuration_type: configType,
            integration_time: parseFloat(document.getElementById('integration-time').value),
            time_step: parseFloat(document.getElementById('time-step').value),
            size_factor: parseFloat(document.getElementById('size-factor').value),
            G: parseFloat(document.getElementById('gravitational-constant').value)
        };
        
        if (configType === 'homothetic') {
            parameters.velocity_factor = parseFloat(document.getElementById('velocity-factor').value);
        } else if (configType === 'lagrangian') {
            const rotationRate = document.getElementById('rotation-rate').value;
            parameters.rotation_rate = rotationRate === 'null' ? null : parseFloat(rotationRate);
        } else if (configType === 'random') {
            parameters.random_seed = 42;
        } else if (configType === 'user_drawn') {
            // Try to parse textarea first
            let initialStateStr = document.getElementById('user-drawn-initial-state').value;
            let initialState = [];
            try {
                if (initialStateStr.trim().length > 0) {
                    initialState = JSON.parse(initialStateStr);
                }
            } catch (e) {
                alert('Invalid JSON for initial state.');
            }
            // If file is uploaded, override
            const fileInput = document.getElementById('user-drawn-file');
            if (fileInput.files && fileInput.files[0]) {
                // File reading is async, so we handle it in runSimulation
                parameters._user_drawn_file = fileInput.files[0];
            }
            parameters.initial_state = initialState;
        }
        return parameters;
    }
    
    async runSimulation() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.updateStatus('Running simulation...', 'running');
        this.setLoadingState(true);
        try {
            let parameters = this.getSimulationParameters();
            // If user_drawn and file is uploaded, read file
            if (parameters.configuration_type === 'user_drawn' && parameters._user_drawn_file) {
                const file = parameters._user_drawn_file;
                const text = await file.text();
                try {
                    parameters.initial_state = JSON.parse(text);
                } catch (e) {
                    alert('Invalid JSON in uploaded file.');
                    this.isRunning = false;
                    this.setLoadingState(false);
                    return;
                }
                delete parameters._user_drawn_file;
            }
            const response = await fetch('/api/simulate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ parameters })
            });
            const data = await response.json();
            if (data.success) {
                this.currentResults = data.results;
                this.updateVisualizations();
                this.updateSimulationInfo(data.results.metadata);
                this.updateStatus('Simulation completed', 'ready');
            } else {
                throw new Error(data.error || 'Simulation failed');
            }
        } catch (error) {
            console.error('Simulation error:', error);
            this.updateStatus('Simulation failed', 'error');
        } finally {
            this.isRunning = false;
            this.setLoadingState(false);
        }
    }
    
    resetSimulation() {
        // Reset form to default values
        document.getElementById('config-type').value = 'homothetic';
        document.getElementById('mass1').value = '1.0';
        document.getElementById('mass2').value = '1.0';
        document.getElementById('mass3').value = '1.0';
        document.getElementById('integration-time').value = '10';
        document.getElementById('integration-time-value').textContent = '10s';
        document.getElementById('time-step').value = '0.01';
        document.getElementById('time-step-value').textContent = '0.01s';
        document.getElementById('size-factor').value = '1.0';
        document.getElementById('size-factor-value').textContent = '1.0';
        document.getElementById('velocity-factor').value = '0.0';
        document.getElementById('velocity-factor-value').textContent = '0.0';
        document.getElementById('rotation-rate').value = '1.0';
        document.getElementById('rotation-rate-value').textContent = '1.0';
        document.getElementById('gravitational-constant').value = '1.0';
        document.getElementById('preset-select').value = '';
        
        this.updateConfigurationUI('homothetic');
        this.clearPlots();
        this.updateStatus('Ready', 'ready');
    }
    
    updateStatus(message, type) {
        const statusIndicator = document.getElementById('status-indicator');
        statusIndicator.textContent = message;
        statusIndicator.className = `navbar-text ${type}`;
    }
    
    setLoadingState(loading) {
        const runButton = document.getElementById('run-simulation');
        if (loading) {
            runButton.disabled = true;
            runButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running...';
        } else {
            runButton.disabled = false;
            runButton.innerHTML = '<i class="fas fa-play"></i> Run Simulation';
        }
    }
    
    initializePlots() {
        // Initialize 3D trajectory plot
        const trajectoryData = [{
            type: 'scatter3d',
            mode: 'lines',
            x: [],
            y: [],
            z: [],
            line: { width: 2 },
            name: 'Body 1'
        }, {
            type: 'scatter3d',
            mode: 'lines',
            x: [],
            y: [],
            z: [],
            line: { width: 2 },
            name: 'Body 2'
        }, {
            type: 'scatter3d',
            mode: 'lines',
            x: [],
            y: [],
            z: [],
            line: { width: 2 },
            name: 'Body 3'
        }];
        
        const trajectoryLayout = {
            title: 'Three-Body Problem Trajectories',
            scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Z' }
            },
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        Plotly.newPlot('trajectory-plot', trajectoryData, trajectoryLayout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
        });
        
        // Initialize energy conservation plot
        const energyData = [{
            type: 'scatter',
            mode: 'lines',
            x: [],
            y: [],
            name: 'Total Energy',
            line: { color: '#007bff' }
        }];
        
        const energyLayout = {
            title: 'Energy Conservation',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Energy' },
            margin: { l: 50, r: 20, t: 50, b: 50 }
        };
        
        Plotly.newPlot('energy-plot', energyData, energyLayout, {
            responsive: true,
            displayModeBar: false
        });
        
        // Initialize angular momentum plot
        const angularMomentumData = [{
            type: 'scatter',
            mode: 'lines',
            x: [],
            y: [],
            name: 'Angular Momentum Magnitude',
            line: { color: '#28a745' }
        }];
        
        const angularMomentumLayout = {
            title: 'Angular Momentum Conservation',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Angular Momentum' },
            margin: { l: 50, r: 20, t: 50, b: 50 }
        };
        
        Plotly.newPlot('angular-momentum-plot', angularMomentumData, angularMomentumLayout, {
            responsive: true,
            displayModeBar: false
        });
    }
    
    updateVisualizations() {
        if (!this.currentResults) return;
        
        // Update 3D trajectory plot
        const trajectoryData = this.currentResults.bodies.map((body, index) => ({
            type: 'scatter3d',
            mode: 'lines',
            x: body.positions.map(pos => pos[0]),
            y: body.positions.map(pos => pos[1]),
            z: body.positions.map(pos => pos[2]),
            line: { 
                width: 3,
                color: ['#ff6b6b', '#4ecdc4', '#45b7d1'][index]
            },
            name: body.name
        }));
        
        const trajectoryLayout = {
            title: `Three-Body Problem Trajectories (σ = ${this.currentResults.metadata.sigma.toFixed(6)})`,
            scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Z' }
            },
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        Plotly.react('trajectory-plot', trajectoryData, trajectoryLayout);
        
        // Update energy conservation plot
        const times = this.currentResults.times;
        const energyData = [{
            type: 'scatter',
            mode: 'lines',
            x: times,
            y: this.currentResults.energy,
            name: 'Total Energy',
            line: { color: '#007bff' }
        }];
        
        const energyLayout = {
            title: 'Energy Conservation',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Energy' },
            margin: { l: 50, r: 20, t: 50, b: 50 }
        };
        
        Plotly.react('energy-plot', energyData, energyLayout);
        
        // Update angular momentum plot
        const angularMomentumData = [{
            type: 'scatter',
            mode: 'lines',
            x: times,
            y: this.currentResults.angular_momentum,
            name: 'Angular Momentum Magnitude',
            line: { color: '#28a745' }
        }];
        
        const angularMomentumLayout = {
            title: 'Angular Momentum Conservation',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Angular Momentum' },
            margin: { l: 50, r: 20, t: 50, b: 50 }
        };
        
        Plotly.react('angular-momentum-plot', angularMomentumData, angularMomentumLayout);
        
        this.updatePoincarePlot();
        this.updatePhasePlot();
    }
    
    updateSimulationInfo(metadata) {
        document.getElementById('sigma-value').textContent = metadata.sigma.toFixed(6);
        document.getElementById('config-value').textContent = metadata.configuration_type;
        document.getElementById('energy-value').textContent = 'Calculated';
        document.getElementById('angular-momentum-value').textContent = 'Calculated';
    }
    
    clearPlots() {
        // Clear all plots
        Plotly.react('trajectory-plot', [], {});
        Plotly.react('energy-plot', [], {});
        Plotly.react('angular-momentum-plot', [], {});
        
        // Reset simulation info
        document.getElementById('sigma-value').textContent = '-';
        document.getElementById('config-value').textContent = '-';
        document.getElementById('energy-value').textContent = '-';
        document.getElementById('angular-momentum-value').textContent = '-';
    }

    async updatePoincarePlot() {
        if (!this.currentResults) return;
        const body = document.getElementById('poincare-body').value;
        const plane = document.getElementById('poincare-plane').value;
        const response = await fetch('/api/poincare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: this.currentResults, body, plane })
        });
        const data = await response.json();
        const trace = {
            x: data.x,
            y: data.vx,
            mode: 'markers',
            type: 'scatter',
            marker: { color: '#e67e22', size: 6 },
        };
        Plotly.newPlot('poincare-plot', [trace], {
            title: 'Poincaré Section',
            xaxis: { title: plane === 'y' ? 'x' : 'y' },
            yaxis: { title: plane === 'y' ? 'vx' : 'vy' },
            margin: { l: 50, r: 20, t: 50, b: 50 }
        }, { responsive: true, displayModeBar: false });
    }

    async updatePhasePlot() {
        if (!this.currentResults) return;
        const body = document.getElementById('phase-body').value;
        const xvar = document.getElementById('phase-x').value;
        const vvar = document.getElementById('phase-vx').value;
        const response = await fetch('/api/phase', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: this.currentResults, body, xvar, vvar })
        });
        const data = await response.json();
        const trace = {
            x: data.x,
            y: data.vx,
            mode: 'lines',
            type: 'scatter',
            line: { color: '#16a085' },
        };
        Plotly.newPlot('phase-plot', [trace], {
            title: 'Phase Space',
            xaxis: { title: xvar },
            yaxis: { title: vvar },
            margin: { l: 50, r: 20, t: 50, b: 50 }
        }, { responsive: true, displayModeBar: false });
    }

    async downloadData() {
        if (!this.currentResults) return;
        const format = 'csv'; // or 'json' if you want to add a selector
        const response = await fetch('/api/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ results: this.currentResults, format })
        });
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = format === 'csv' ? 'three_body_simulation.csv' : 'three_body_simulation.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.visualizer = new ThreeBodyVisualizer();
}); 