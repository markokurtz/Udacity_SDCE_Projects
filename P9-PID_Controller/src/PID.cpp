#include "PID.h"

using namespace std;

/*
 * TODO: Complete the PID class.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  PID::Kp = Kp;
  PID::Ki = Ki;
  PID::Kd = Kd;
}

void PID::UpdateError(double cte) {
  // pid implementation from SDC Pid Control lesson16
  double diff_cte = cte - prev_cte;
  prev_cte = cte;
  int_cte += cte;

  p_error = -Kp * cte;
  i_error = -Ki * int_cte;
  d_error = -Kd * diff_cte;
}

double PID::TotalError() { return p_error + d_error + i_error; }
