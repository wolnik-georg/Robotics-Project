Complete Slide-by-Slide Explanation Guide 🎯
Slide 1: Overview
What this slide is about: This slide establishes your research motivation and the three key questions that drive your entire investigation into acoustic sensing for geometric reconstruction.

What you should explain: You should explain that your goal is to extract maximum geometric information from acoustic signals, not just basic contact detection. You're asking three fundamental questions: First, whether you can replicate the classification results that other researchers have achieved with your current experimental setup. Second, what specific types of geometric information can be extracted from acoustic signals beyond simple contact detection. Third, which acoustic features are most important for making accurate classifications, which will help optimize your system for practical applications.

Slide 2: Data and Features
What this slide is about: This slide presents the experimental methodology and data collection approach that forms the foundation of your research, including the dataset size, signal types, and feature extraction strategy.

What you should explain: You should explain that you conducted four separate experimental batches with 650 total samples, ensuring 50 samples per class for balanced machine learning training. You used 2-second broadband chirp signals covering the full audible spectrum from 20Hz to 20kHz to test the complete acoustic response. You tested three different classification tasks: contact position classification to determine where on the finger contact occurs, edge detection to distinguish geometric edges from flat surfaces, and fine feature detection to identify small metal objects. From each audio recording, you extracted 53 features total: 38 standard acoustic features analyzing spectral, temporal, and frequency domain properties, plus 15 advanced impulse response features from transfer function analysis.

Slide 3: Acoustic Features - Spectral
What this slide is about: This slide details the first category of acoustic features you extract, focusing on standard audio processing techniques adapted for acoustic sensing applications.

What you should explain: You should explain that these are well-established audio signal processing features adapted for acoustic sensing. The spectral features like spectral centroid measure the center of mass of the frequency spectrum, while spectral bandwidth measures how spread out the frequencies are around that center. These features are sensitive to how different contact conditions change the frequency distribution of the acoustic signal. The temporal features capture how the signal changes over time, with zero crossing rate indicating signal noisiness and envelope features describing amplitude properties. These features are based on standard methods from audio classification literature but are applied to the novel domain of robotic tactile sensing.

Slide 4: Acoustic Features - MFCC and High-Frequency
What this slide is about: This slide covers two additional categories of acoustic features: MFCC coefficients from speech processing and custom high-frequency features designed for material property detection.

What you should explain: You should explain that MFCC features are standard in speech processing and provide a perceptually relevant representation of spectral content that captures timbre and spectral envelope information. The high-frequency features focus on energy above 8kHz, which is particularly sensitive to material properties and surface texture because high frequencies are affected by contact deformation and surface roughness. These features are important because different materials and geometric conditions create distinct signatures in the high-frequency range, making them valuable for material classification and fine geometric discrimination.

Slide 5: Impulse Response Features - Magnitude
What this slide is about: This slide introduces the advanced transfer function features that characterize the acoustic system's frequency response properties independent of the input signal.

What you should explain: You should explain that these features come from transfer function analysis where you compute H(f) = Output/Input to get the acoustic fingerprint of each contact condition. These features describe properties like the center of mass and spread of the frequency response, the strongest resonance characteristics, and the asymmetry of resonance distributions. The key advantage is that these features are independent of the specific input signal you use - they characterize the physical acoustic system itself, providing complementary information to traditional acoustic features that depend on both the system and the input signal properties.

Slide 6: Impulse Response Features - Decay and Damping
What this slide is about: This slide covers the temporal and damping characteristics extracted from the impulse response, focusing on how the system responds and decays over time.

What you should explain: You should explain that these features capture how the acoustic system responds over time, including amplitude decay rates, exponential decay time constants, and system damping measures. The quality factor (Q-factor) measures resonance sharpness, which changes with different contact conditions. These features are valuable because they capture physical properties like material stiffness, contact dynamics, and damping characteristics that are fundamental to understanding the geometric and material properties of contacted objects. They provide true system characterization that complements the frequency domain features.

Slide 7: Dimensionality Reduction Methods
What this slide is about: This slide explains the mathematical techniques used to visualize your high-dimensional feature space and assess whether different classes can be separated.

What you should explain: You should explain that you're using two complementary dimensionality reduction techniques to visualize your 53-dimensional feature space in 2D. PCA is a linear technique that finds the directions of maximum variance and preserves global structure and distances between data points. t-SNE is a nonlinear technique that preserves local neighborhoods and is excellent for revealing cluster structure in complex, high-dimensional data. The purpose is to reduce your 53 features down to 2 dimensions so you can visually assess whether your features contain enough discriminative information to separate the different classes.

Slide 8: PCA and t-SNE Results
What this slide is about: This slide presents the visual evidence that your features successfully separate different contact classes, proving that the acoustic signals contain discriminative information for geometric classification.

What you should explain: You should explain that this visualization provides direct evidence that your features work for classification. The PCA plot on the left shows that classes are linearly separable, meaning simple linear classifiers should work well. The t-SNE plot on the right reveals nonlinear cluster structure, showing that classes form distinct groups in feature space. The clear separation between different colored clusters indicates that your 53 acoustic features contain sufficient discriminative information to distinguish between different contact conditions, validating your feature extraction approach.

Slide 9: Transfer Function - Input Signal
What this slide is about: This slide explains the controlled input signal you use for system identification, describing the properties of your broadband chirp signal and why this approach is effective.

What you should explain: You should explain that you send a carefully designed broadband chirp signal that sweeps linearly from 20Hz to 20kHz over 2 seconds. This signal has constant amplitude across all frequencies and is designed to excite the entire frequency response of the acoustic system. Using a known, controlled input signal is essential for system identification because it allows you to characterize how the system modifies different frequencies. The 2-second duration provides good frequency resolution for analysis, and the broadband nature ensures you test the complete acoustic response of the finger-object interaction.

Slide 10: Transfer Function - Received Signal
What this slide is about: This slide describes what happens to your input signal when it interacts with the finger-object system, explaining how different contact conditions modify the acoustic response.

What you should explain: You should explain that when your controlled input signal travels through the finger and interacts with contacted objects, it gets modified in ways that depend on the specific contact conditions. The microphone captures this modified response, which contains information about the geometric and material properties of the contact. The key observation is that different contact conditions - whether touching at different positions, contacting edges versus flat surfaces, or touching different materials - produce distinctly different acoustic responses. This modified response contains rich information that can be extracted as features for classification.

Slide 11: Transfer Function - Calculation
What this slide is about: This slide explains the mathematical process of deconvolution used to extract the transfer function from your input and output signals.

What you should explain: You should explain the three-step process for computing the transfer function. First, you take the Fourier transform of both your input sweep signal and the recorded response to convert them to the frequency domain. Second, you perform deconvolution by dividing the output spectrum by the input spectrum: H(f) = Y(f)/X(f). Third, the result is a transfer function that shows exactly how the system modifies each frequency component. The key advantage is that this transfer function is independent of your specific input signal - it characterizes the acoustic properties of the physical system itself, providing a pure description of how the finger-object interaction affects sound.

Slide 12: Transfer Function - Additional Features
What this slide is about: This slide summarizes the value of transfer function analysis and explains how it adds 15 new features to complement your existing acoustic features.

What you should explain: You should explain that transfer function analysis gives you access to 15 new impulse response features that describe resonance characteristics, system damping properties, and frequency response statistics. These features are particularly valuable because traditional acoustic features depend on your input signal design, while impulse response features characterize the physical system properties independent of the input. This provides complementary information that enhances your classification performance. The combination of 38 acoustic features plus 15 impulse response features gives you 53 total features for training your machine learning models.

Slide 13: Saliency Analysis - Method
What this slide is about: This slide explains the methodology for determining which of your 53 features are most important for classification decisions.

What you should explain: You should explain that saliency analysis helps you understand which features most influence your neural network's classification decisions. You train a neural network on your complete dataset of 650 samples with all 53 features, then use backpropagation to calculate gradients with respect to input features. Features with higher gradient magnitudes have more influence on the network's predictions, indicating greater importance for classification. This analysis is implemented using TensorFlow/Keras with multiple hidden layers to capture complex feature interactions, providing scientific insight into which acoustic properties contain the most geometric information.

Slide 14: Saliency Analysis - Results
What this slide is about: This slide presents the specific results of your feature importance analysis, identifying the top 6 most discriminative features and their relative importance.

What you should explain: You should explain that your saliency analysis identified six key features as most important for classification. The top features include spectral bandwidth (frequency spread), resonance skewness (resonance asymmetry), frequency response centroid (response center), ultra-high energy ratio (high-frequency energy), decay amplitude (signal decay rate), and ultra-high ratio (high-frequency ratio). Importantly, three of the top six features are derived from impulse response analysis, showing that both acoustic and transfer function features contribute significantly. The mixed importance between feature types demonstrates that no single approach dominates - the combination provides the best performance.

Slide 15: Saliency Analysis - Insights
What this slide is about: This slide interprets the results of your saliency analysis, explaining what these findings mean scientifically and practically for acoustic sensing applications.

What you should explain: You should explain that these results provide both scientific and practical insights. Scientifically, they show that impulse response features are among the most discriminative for geometric classification, validating your transfer function approach. At the same time, acoustic features provide crucial complementary information, confirming that the combination of both feature types is superior to either alone. Practically, these results show that you can achieve high performance (around 95% accuracy) with just the top 6 features, enabling computational optimization for real-time applications by focusing on the most important acoustic properties.

Slide 16: Conclusions
What this slide is about: This slide summarizes your main findings and their significance for acoustic sensing research and applications.

What you should explain: You should explain that your research demonstrates three key findings. First, PCA and t-SNE visualization confirm that your features provide clear class separability, validating your approach to acoustic feature extraction. Second, transfer function analysis adds valuable impulse response features that complement traditional acoustic features and improve classification performance. Third, saliency analysis successfully identifies the most important features, enabling both scientific understanding of which acoustic properties matter most and practical optimization for computational efficiency.

Slide 17: Next Steps
What this slide is about: This slide outlines the future research directions and practical applications that build on your current findings.

What you should explain: You should explain that your next steps involve integrating your acoustic sensing pipeline with robot control systems to enable real-time geometric reconstruction and classification. You plan to expand your experiments to test the limits of acoustic sensing performance and explore how to optimize input signals to extract maximum information. Key questions for future work include determining the optimal signal design for different sensing tasks and systematically testing the boundaries of what geometric information can be reliably extracted through acoustic methods. This work will advance acoustic sensing from research validation toward practical robotic applications.