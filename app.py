#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Three-Body Problem Visualization Application

A Flask web application for interactive visualization of three-body problem
configurations with real-time parameter adjustment capabilities.
"""

import os
import json
import numpy as np
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
    make_response,
)
from flask_cors import CORS
import threading
import time
from datetime import datetime
import io
import csv

# Import our three-body problem modules
from three_body_problem import ThreeBodyProblem, HomotheticOrbits, LagrangianSolutions
from quaternion import Quaternion

app = Flask(__name__)
CORS(app)

# Global variables for simulation state
simulation_state = {
    "running": False,
    "current_results": None,
    "simulation_thread": None,
    "parameters": {
        "masses": [1.0, 1.0, 1.0],
        "configuration_type": "homothetic",  # 'homothetic', 'lagrangian', 'custom'
        "integration_time": 10.0,
        "time_step": 0.01,
        "size_factor": 1.0,
        "velocity_factor": 0.0,
        "rotation_rate": None,
        "G": 1.0,
    },
}


class RealTimeSimulator:
    """Handles real-time simulation of three-body problem"""

    def __init__(self):
        self.tbp = None
        self.results = None
        self.is_running = False

    def setup_simulation(self, parameters):
        """Setup the three-body problem with given parameters"""
        masses = np.array(parameters["masses"])
        self.tbp = ThreeBodyProblem(masses, G=parameters["G"])

        if parameters["configuration_type"] == "homothetic":
            homothetic = HomotheticOrbits(self.tbp)
            initial_state = homothetic.generate_initial_state(
                size=parameters["size_factor"],
                velocity_factor=parameters["velocity_factor"],
            )
        elif parameters["configuration_type"] == "lagrangian":
            lagrangian = LagrangianSolutions(self.tbp)
            initial_state = lagrangian.generate_initial_state(
                size=parameters["size_factor"],
                rotation_rate=parameters["rotation_rate"],
            )
        elif parameters["configuration_type"] == "free_fall":
            size = parameters["size_factor"]
            a1 = np.array([size, 0, 0])
            a2 = np.array([-0.5 * size, 0.866 * size, 0])
            a3 = np.array([-0.5 * size, -0.866 * size, 0])
            cm = (masses[0] * a1 + masses[1] * a2 + masses[2] * a3) / np.sum(masses)
            r1 = a1 - cm
            r2 = a2 - cm
            r3 = a3 - cm
            p1 = np.zeros(3)
            p2 = np.zeros(3)
            p3 = np.zeros(3)
            initial_state = np.concatenate([r1, r2, r3, p1, p2, p3])
        elif parameters["configuration_type"] == "collinear":
            size = parameters["size_factor"]
            spacing = size
            r1 = np.array([-spacing, 0, 0])
            r2 = np.array([0, 0, 0])
            r3 = np.array([spacing, 0, 0])
            p1 = np.zeros(3)
            p2 = np.zeros(3)
            p3 = np.zeros(3)
            initial_state = np.concatenate([r1, r2, r3, p1, p2, p3])
        elif parameters["configuration_type"] == "random":
            size = parameters["size_factor"]
            rng = np.random.default_rng(parameters.get("random_seed", None))
            r = rng.uniform(-size, size, (3, 3))
            v = rng.uniform(-1, 1, (3, 3))
            p = v * masses[:, None]
            initial_state = np.concatenate([r[0], r[1], r[2], p[0], p[1], p[2]])
        elif parameters["configuration_type"] == "user_drawn":
            initial_state = np.array(parameters["initial_state"])
        else:  # custom
            size = parameters["size_factor"]
            initial_state = np.array(
                [
                    size * np.cos(0),
                    size * np.sin(0),
                    0,  # r1
                    size * np.cos(2 * np.pi / 3),
                    size * np.sin(2 * np.pi / 3),
                    0,  # r2
                    size * np.cos(4 * np.pi / 3),
                    size * np.sin(4 * np.pi / 3),
                    0,  # r3
                    0,
                    0,
                    0,  # p1
                    0,
                    0,
                    0,  # p2
                    0,
                    0,
                    0,  # p3
                ]
            )
        return initial_state

    def run_simulation(self, parameters):
        """Run the simulation with given parameters"""
        try:
            initial_state = self.setup_simulation(parameters)

            # Integrate the equations of motion
            t_span = (0, parameters["integration_time"])
            t_eval = np.arange(
                0, parameters["integration_time"], parameters["time_step"]
            )

            if self.tbp is None:
                raise ValueError("Three-body problem not initialized")

            self.results = self.tbp.integrate(
                initial_state=initial_state,
                t_span=t_span,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )

            # Add additional information
            self.results["masses"] = parameters["masses"]
            self.results["sigma"] = self.tbp.sigma
            self.results["configuration_type"] = parameters["configuration_type"]
            self.results["parameters"] = parameters

            return self.results

        except Exception as e:
            print(f"Simulation error: {e}")
            return None


# Initialize the simulator
simulator = RealTimeSimulator()


@app.route("/")
def index():
    """Main page"""
    return render_template("index.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run a simulation with given parameters"""
    global simulation_state

    try:
        data = request.get_json()
        parameters = data.get("parameters", simulation_state["parameters"])

        # Update global parameters
        simulation_state["parameters"].update(parameters)

        # Run simulation
        results = simulator.run_simulation(simulation_state["parameters"])

        if results is None:
            return jsonify({"error": "Simulation failed"}), 400

        # Format results for frontend
        formatted_results = format_simulation_results(results)

        return jsonify(
            {
                "success": True,
                "results": formatted_results,
                "parameters": simulation_state["parameters"],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/parameters", methods=["GET"])
def get_parameters():
    """Get current simulation parameters"""
    return jsonify(simulation_state["parameters"])


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Get predefined simulation presets"""
    presets = {
        "equal_masses_homothetic": {
            "name": "Equal Masses - Homothetic",
            "masses": [1.0, 1.0, 1.0],
            "configuration_type": "homothetic",
            "integration_time": 10.0,
            "size_factor": 1.0,
            "velocity_factor": 0.0,
        },
        "equal_masses_lagrangian": {
            "name": "Equal Masses - Lagrangian",
            "masses": [1.0, 1.0, 1.0],
            "configuration_type": "lagrangian",
            "integration_time": 10.0,
            "size_factor": 1.0,
            "rotation_rate": None,
        },
        "figure_eight": {
            "name": "Figure-Eight Solution",
            "masses": [1.0, 1.0, 1.0],
            "configuration_type": "custom",
            "integration_time": 15.0,
            "size_factor": 1.0,
        },
        "exceptional_sigma_1_3": {
            "name": "Exceptional σ = 1/3",
            "masses": [1.0, 1.0, 1.0],  # This gives σ = 1/3
            "configuration_type": "homothetic",
            "integration_time": 12.0,
            "size_factor": 1.0,
            "velocity_factor": 0.1,
        },
        "heavy_central_body": {
            "name": "Heavy Central Body",
            "masses": [10.0, 1.0, 1.0],
            "configuration_type": "lagrangian",
            "integration_time": 8.0,
            "size_factor": 1.0,
            "rotation_rate": 1.0,
        },
        "free_fall": {
            "name": "Free Fall (Equilateral, Zero Velocity)",
            "masses": [1.0, 1.0, 1.0],
            "configuration_type": "free_fall",
            "integration_time": 10.0,
            "size_factor": 1.0,
        },
        "collinear": {
            "name": "Collinear (x-axis, Zero Velocity)",
            "masses": [1.0, 1.0, 1.0],
            "configuration_type": "collinear",
            "integration_time": 10.0,
            "size_factor": 1.0,
        },
        "random": {
            "name": "Random Initial Conditions",
            "masses": [1.0, 1.0, 1.0],
            "configuration_type": "random",
            "integration_time": 10.0,
            "size_factor": 1.0,
            "random_seed": 42,
        },
        "user_drawn": {
            "name": "User-Drawn/Uploaded",
            "masses": [1.0, 1.0, 1.0],
            "configuration_type": "user_drawn",
            "integration_time": 10.0,
            "size_factor": 1.0,
            "initial_state": [0] * 18,
        },
    }
    return jsonify(presets)


def format_simulation_results(results):
    """Format simulation results for frontend consumption, including energy and angular momentum time series."""
    states = results["states"]
    times = results["t"]

    # Extract positions for each body
    r1 = states[:, 0:3]
    r2 = states[:, 3:6]
    r3 = states[:, 6:9]

    # Extract velocities for each body
    p1 = states[:, 9:12]
    p2 = states[:, 12:15]
    p3 = states[:, 15:18]

    masses = results.get("masses", [1.0, 1.0, 1.0])

    # Calculate velocities from momenta
    v1 = p1 / masses[0]
    v2 = p2 / masses[1]
    v3 = p3 / masses[2]

    # Compute Hamiltonian (energy) and angular momentum time series
    tbp = simulator.tbp
    if tbp is None:
        # Fallback: create a new instance if for some reason simulator.tbp is not set
        tbp = ThreeBodyProblem(
            np.array(masses), G=results.get("parameters", {}).get("G", 1.0)
        )
    energy_series = []
    angular_momentum_series = []
    for state in states:
        energy_series.append(tbp.hamiltonian(state))
        L = tbp.angular_momentum(state)
        angular_momentum_series.append(np.linalg.norm(L))

    formatted = {
        "times": times.tolist(),
        "bodies": [
            {
                "name": f"Body 1 (m={masses[0]:.2f})",
                "positions": r1.tolist(),
                "velocities": v1.tolist(),
                "mass": masses[0],
            },
            {
                "name": f"Body 2 (m={masses[1]:.2f})",
                "positions": r2.tolist(),
                "velocities": v2.tolist(),
                "mass": masses[1],
            },
            {
                "name": f"Body 3 (m={masses[2]:.2f})",
                "positions": r3.tolist(),
                "velocities": v3.tolist(),
                "mass": masses[2],
            },
        ],
        "energy": energy_series,
        "angular_momentum": angular_momentum_series,
        "metadata": {
            "sigma": results.get("sigma", "unknown"),
            "configuration_type": results.get("configuration_type", "unknown"),
            "total_time": times[-1],
            "num_points": len(times),
            "conservation_errors": results.get("conservation_errors", {}),
            "collisions": results.get("collisions", {}),
        },
    }

    return formatted


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files"""
    return send_from_directory("static", filename)


@app.route("/api/poincare", methods=["POST"])
def poincare_section():
    data = request.get_json()
    results = data["results"]
    body_idx = int(data.get("body", 0))
    plane = data.get("plane", "y")
    # Extract positions and velocities
    positions = np.array(results["bodies"][body_idx]["positions"])
    velocities = np.array(results["bodies"][body_idx]["velocities"])
    times = np.array(results["times"])
    # Find crossings
    if plane == "y":
        y = positions[:, 1]
        vy = velocities[:, 1]
        mask = (y[:-1] * y[1:] < 0) & (vy[1:] > 0)
        x_cross = positions[1:, 0][mask]
        vx_cross = velocities[1:, 0][mask]
        poincare_x = x_cross
        poincare_vx = vx_cross
    else:
        x = positions[:, 0]
        vx = velocities[:, 0]
        mask = (x[:-1] * x[1:] < 0) & (vx[1:] > 0)
        y_cross = positions[1:, 1][mask]
        vy_cross = velocities[1:, 1][mask]
        poincare_x = y_cross
        poincare_vx = vy_cross
    return jsonify({"x": poincare_x.tolist(), "vx": poincare_vx.tolist()})


@app.route("/api/phase", methods=["POST"])
def phase_space():
    data = request.get_json()
    results = data["results"]
    body_idx = int(data.get("body", 0))
    xvar = data.get("xvar", "x")
    vvar = data.get("vvar", "vx")
    positions = np.array(results["bodies"][body_idx]["positions"])
    velocities = np.array(results["bodies"][body_idx]["velocities"])
    idx_map = {"x": 0, "y": 1, "z": 2, "vx": 0, "vy": 1, "vz": 2}
    xdata = positions[:, idx_map[xvar]]
    vdata = velocities[:, idx_map[vvar]]
    return jsonify({"x": xdata.tolist(), "vx": vdata.tolist()})


@app.route("/api/download", methods=["POST"])
def download_data():
    data = request.get_json()
    results = data["results"]
    fmt = data.get("format", "csv")
    output = io.StringIO()
    if fmt == "csv":
        writer = csv.writer(output)
        header = ["time"]
        for i in range(3):
            header += [
                f"r{i+1}_x",
                f"r{i+1}_y",
                f"r{i+1}_z",
                f"v{i+1}_x",
                f"v{i+1}_y",
                f"v{i+1}_z",
            ]
        header += ["energy", "angular_momentum"]
        writer.writerow(header)
        for i, t in enumerate(results["times"]):
            row = [t]
            for b in range(3):
                row += results["bodies"][b]["positions"][i]
                row += results["bodies"][b]["velocities"][i]
            row.append(results["energy"][i])
            row.append(results["angular_momentum"][i])
            writer.writerow(row)
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = (
            "attachment; filename=three_body_simulation.csv"
        )
        response.headers["Content-Type"] = "text/csv"
        return response
    else:
        response = make_response(json.dumps(results))
        response.headers["Content-Disposition"] = (
            "attachment; filename=three_body_simulation.json"
        )
        response.headers["Content-Type"] = "application/json"
        return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
