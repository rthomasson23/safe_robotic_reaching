This is a project for AA203 at Staford, taken in Spring 2021. 

The following files handle the creation and assessment of the spline trajectory optimizer described in the report/video.
	NominalTrajectory.py
	NominalPlotting.py
	EulerDynamics.py
	AnimateTest.py

q_PD.py is the kinematics following an implementation of the PD controller described in the report/video.

The \scripts folder contains files to run the RRR manipulator in PyBullet using the PD+CBF controller described in the report/video. Run \scripts\simRRR.py to start the simulation. 

URDFs for the robot and obstacle can be found in \RRR_manipulator_description and \obstacle_description, respectively. 
