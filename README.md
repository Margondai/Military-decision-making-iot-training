# Military-decision-making-iot-training

Adaptive Simulation-Based Training for Military Decision-Making: Leveraging IoT-Derived Cognitive and Emotional Feedback

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Conference](https://img.shields.io/badge/Conference-MODSIM%20World%202025-green.svg)](https://modsimworld.org/)

## Abstract

Military personnel are often required to make rapid, high-stakes decisions under intense stress, where cognitive overload and emotional dysregulation can impair executive functioning and increase the risk of mission failure. This project develops a neuroadaptive, closed-loop simulation framework that integrates real-time physiological monitoring to enhance military training. Using wearable IoT devices (including EEG, heart rate monitors, and galvanic skin response sensors), the system continuously assesses cognitive load and emotional stress, dynamically adjusting scenario complexity, pacing, and environmental stimuli to match each trainee's current cognitive and emotional state.

## Overview

Traditional simulation-based education (SBE) systems are limited in their ability to adapt in real time to trainee's mental state, reducing their effectiveness in preparing individuals for dynamic operational environments. This research addresses this gap by developing a neuroadaptive system that uses real-time EEG and biosensor data to drive personalized training adjustments.

### Key Features

- **Multimodal Physiological Monitoring**: Real-time EEG, heart rate variability (HRV), galvanic skin response (GSR), temperature, and accelerometry data
- **Neuroadaptive Framework**: Closed-loop system that adjusts training scenarios based on cognitive load and stress levels
- **Military Decision-Making Scenarios**: Realistic tactical situations including air support deployment, position holding, and communication under pressure
- **Recovery Assessment**: Heart rate variability (RMSSD) tracking for emotional regulation pre- and post-training
- **Statistical Analysis**: Comprehensive APA-style reporting with independent t-tests, effect sizes, and confidence intervals

## System Architecture

The framework operates through a four-stage adaptive cycle:

1. **Data Collection**: Multimodal sensor data from EEG and wearable IoT devices
2. **State Estimation**: Random Forest classifier processes 16 physiological features to assess cognitive load and stress
3. **Adaptive Action Selection**: Dynamic adjustment of tactical scenarios, time pressure, and communication noise
4. **Environment Update**: Real-time modification of training parameters based on physiological feedback

## Results

### Training Effectiveness Metrics

- **Decision Accuracy**: Consistent performance across adaptive (80.0%) and static (80.5%) modes
- **Task Completion Time**: Comparable efficiency between modes (~3.5 seconds average)
- **Physiological Recovery**: Significant RMSSD improvements in both modes (adaptive: 14.34→15.77, static: 13.05→14.36)
- **Learning Retention**: Enhanced retention in adaptive mode (2.05% vs 0.18% improvement)
- **Cognitive Failure Rate**: 0% in both modes, indicating effective stress management

The adaptive mode demonstrated superior physiological recovery and retention outcomes while maintaining decision accuracy and efficiency.

## Installation

### Prerequisites

- Python 3.8 or higher
- Required dependencies listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/yourusername/military-decision-making-iot-training.git
cd military-decision-making-iot-training
pip install -r requirements.txt
python military_training_system.py
```

## Usage

### Basic Usage

Run the adaptive military training simulation:

```bash
python military_training_system.py
```

The system will automatically:
- Load real EEG data from available datasets or use simulated patterns
- Initialize wearable sensor data processing
- Run both adaptive and static training modes for comparison
- Provide comprehensive statistical analysis with APA-style reporting
- Display real-time VR interface with military scenarios

### System Requirements

- **Pygame Window**: Interactive military scenario visualization
- **Training Duration**: Approximately 10-15 minutes per mode (adaptive + static)
- **Data Processing**: Real-time physiological signal analysis and machine learning classification
- **Recovery Games**: Periodic physiological recovery assessment through interactive target games

## Methodology

### Physiological Data Sources

- **EEG Signals**: Theta, alpha, and beta power from channels T7, F8, Cz, P4
- **Heart Rate Variability**: RMSSD calculation for stress recovery assessment
- **Galvanic Skin Response**: Emotional arousal and stress indicators
- **Temperature & Accelerometry**: Additional physiological state markers

### Machine Learning Architecture

- **Random Forest Classifier**: 16-feature input (11 EEG + 5 wearable sensor features)
- **Real-time Classification**: Continuous cognitive load and stress level estimation
- **Adaptive Logic**: Threshold-based triggers for dynamic scenario adjustment

### Statistical Analysis

The system includes comprehensive statistical evaluation:
- Independent samples t-tests with effect size calculation (Cohen's d)
- Mann-Whitney U tests for non-parametric data
- Pearson and Spearman correlation analyses
- 95% confidence intervals for all metrics
- APA-style reporting format

## Results Summary

### Key Findings

- **Operational Readiness**: Diverse cognitive load profiles (medium: 6507, high: 4689, low: 4164 instances)
- **Individual Variability**: Balanced stress profiles (average: 6675, resilient: 4437, stress-prone: 4248)
- **Physiological Adaptation**: Significant pre-to-post recovery improvements in both training modes
- **Learning Enhancement**: Adaptive mode showed 2.05% retention improvement vs 0.18% in static mode

## File Structure

```
military-decision-making-iot-training/
├── military_training_system.py    # Main application (contains all code)
├── README.md                      # Project documentation
├── LICENSE                        # MIT license
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
└── .gitignore                     # Git ignore rules
```

## Dataset Sources

- **EEG Data**: Motor Movement/Imagery Dataset from OpenNeuro (ds004362)
- **Wearable Data**: Wearable Exam Stress Dataset for physiological monitoring
- **Preprocessing**: 109 subjects with randomized selection for enhanced variability

## Testing

The system includes built-in statistical analysis and validation:

```bash
# The comprehensive analysis runs automatically
python military_training_system.py
```

Statistical outputs include:
- Descriptive statistics for all measures
- Comparative analysis between adaptive and static modes
- Effect size calculations and significance testing
- APA-style formatted results

## Contributing

This is an academic research project. For contributions or collaborations, please contact the authors directly.

## Conference Presentation

This work was presented at MODSIM World 2025. Conference materials including the full paper, presentation slides, and supplementary materials are available in the `conference/` directory.

## Citation

If you use this software in your research, please cite:

```
Acevedo Diaz, A., Margondai, A., Von Ahlefeldt, C., Willox, S., 
Ezcurra, V., Hani, S., Antanavicius, E., Islam, N., & Mouloua, M. (2025). 
Adaptive Simulation-Based Training for Military Decision-Making: 
Leveraging IoT-Derived Cognitive and Emotional Feedback. 
MODSIM World 2025, Orlando, FL.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this research, please contact:

- Anamaria Acevedo Diaz: Anamaria@ucf.edu
- Ancuta Margondai: Ancuta.Margondai@ucf.edu
- Dr. Mustapha Mouloua: Mustapha.Mouloua@ucf.edu

University of Central Florida  
Orlando, Florida
