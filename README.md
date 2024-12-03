## Single View Metrology - Final Project for CS-2467 (Computer Vision)
In this project, I've implemented and evaluated two methods for making scene measurements using a single view of said scene, using a small set of evaluation images I have collected:

- Criminisi's Single View Metrology (height only)
- A Custom Solution using UniDepth, a Monocular Metric Depth Estimation model, and trigonometry to find the lengths of arbitrary lines in the scene.
	- Applied a pose estimation model to try to automatically find 'height-contributing' lines in humans (legs, torso, neck, head) to create a fully automated human height detector, for arbitrary poses

**Note: Evaluating "Single View Metrology in the Wild" was within the project scope as well, but the authors have not released any trained models**