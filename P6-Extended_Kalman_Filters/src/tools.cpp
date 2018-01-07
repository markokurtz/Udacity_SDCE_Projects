#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {

    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {

  MatrixXd Hj(3, 4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  float c1 = px * px + py * py;
  float c2 = sqrt(c1);
  float c3 = (c1 * c2);

  // check division by zero
  if (fabs(c1) < 0.0001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  // compute the Jacobian matrix
  Hj << (px / c2), (py / c2), 0, 0, -(py / c1), (px / c1), 0, 0,
      py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2,
      py / c2;

  return Hj;
}

/*
 * Method normalizes angle to be between -pi and pi interval
 */
double Tools::AngleNormalization(double phi) {
  if (phi > M_PI) {
    return std::fmod(phi + M_PI, 2 * M_PI) - M_PI;
  }
  else if (phi < -M_PI) {
    return std::fmod(phi + M_PI, 2 * M_PI) + M_PI;
  }
  else{
    return phi;}

}

/*
 * Method converts/projects 4d vector cartesian data
 * to 3d vector in polar space
 */
VectorXd Tools::cartesian2polar(VectorXd pred_state) {
	double px = pred_state[0];
	double py = pred_state[1];
	double vx = pred_state[2];
	double vy = pred_state[3];

	VectorXd proj_state = VectorXd(3);
	proj_state << 0,0,0;

	if(px == 0 && py == 0) {
		return proj_state;
	}

  // formulas from lesson 14 - Radar measurements
	double rho = sqrt((px * px) + (py * py));
	double phi = atan2(py, px);
	double rho_dot = ((px * vx) + (py * vy))/rho;

	proj_state[0] = rho;
	proj_state[1] = phi;
	proj_state[2] = rho_dot;

	return proj_state;
}
