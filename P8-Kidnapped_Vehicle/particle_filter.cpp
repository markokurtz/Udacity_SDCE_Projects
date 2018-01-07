/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

// reuse random engine throught functions
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).

  // Set number of particles
  num_particles = 100; // as suggested on udacity QA video for this project

  // code from lesson 15.5
  // normal distributions for x, y and theta/yaw
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[3]);

  // generate particles and insert them in particles vector
  for (int i = 0; i < num_particles; i++) {

    Particle initParticle;

    // initialize particle values from distribution random generator
    initParticle.id = i;
    initParticle.x = dist_x(gen);
    initParticle.y = dist_y(gen);
    initParticle.theta = dist_theta(gen);
    initParticle.weight = 1;

    // add particle to particles vector
    particles.push_back(initParticle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  // normal distribution for sensor noise
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[3]);

  for (int i = 0; i < num_particles; i++) {

    // yawrate is 0 equations
    if (yaw_rate == 0) {
      // kada je 0
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      // no changes in theta

    } else {
      // non zero yawrate equations
      particles[i].x += velocity / yaw_rate *
                        (sin(particles[i].theta + yaw_rate * delta_t) -
                         sin(particles[i].theta));

      particles[i].y += velocity / yaw_rate *
                        (cos(particles[i].theta) -
                         cos(particles[i].theta + yaw_rate * delta_t));

      particles[i].theta += delta_t * yaw_rate;
    }

    // add sensor noise to predicted particle values
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.

  // go through all transformed observations
  for (unsigned int i = 0; i < observations.size(); i++) {

    float obs_x, obs_y;
    obs_x = observations[i].x;
    obs_y = observations[i].y;

    // will keep shortest distance between observation and landmark
    float min_dist = -1;
    int bestMapID = -1;

    // go through all landmarks in sensor range of particle
    for (unsigned int j = 0; j < predicted.size(); j++) {

      float pred_x, pred_y;
      pred_x = predicted[j].x;
      pred_y = predicted[j].y;

      // calculate euclidean distance between particle transfomed observation
      // and map landmark in sensor range
      float distance = dist(obs_x, obs_y, pred_x, pred_y);

      if (min_dist == -1) {
        // first value overwrite min_dist
        min_dist = distance;
        bestMapID = predicted[j].id;
      } else {
        // keep new distances if smaller then previous
        if (distance < min_dist) {
          min_dist = distance;
          bestMapID = predicted[j].id;
        }
      }
    }

    observations[i].id = bestMapID;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems. Keep in mind that this transformation requires
  //   both rotation AND translation (but no scaling). The following is a good
  //   resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html

  // going through all particles
  for (unsigned int i = 0; i < particles.size(); i++) {

    float p_x, p_y, p_theta;
    p_x = particles[i].x;
    p_y = particles[i].y;
    p_theta = particles[i].theta;

    // Transform observations from car data space to map data space
    // using equation 3.33 from http://planning.cs.uiuc.edu/node99.html
    vector<LandmarkObs> transfOBS;

    for (unsigned int j = 0; j < observations.size(); j++) {
      float obs_x, obs_y, trans_x, trans_y;
      int obs_id;
      obs_x = observations[j].x;
      obs_y = observations[j].y;
      obs_id = observations[j].id;

      // transform
      trans_x = (obs_x * cos(p_theta)) - (obs_y * sin(p_theta)) + p_x;
      trans_y = (obs_x * sin(p_theta)) + (obs_y * cos(p_theta)) + p_y;

      LandmarkObs transOB = {obs_id, trans_x, trans_y};

      transfOBS.push_back(transOB);
    }

    // vector of landmarks we will examine for specific particle
    vector<LandmarkObs> landmarksInRange;

    // go through all map landmarks and if they are in sensor range, push to
    // landmarksInRange
    for (unsigned int z = 0; z < map_landmarks.landmark_list.size(); z++) {

      float l_xf, l_yf;
      int l_id;
      l_xf = map_landmarks.landmark_list[z].x_f;
      l_yf = map_landmarks.landmark_list[z].y_f;
      l_id = map_landmarks.landmark_list[z].id_i;

      LandmarkObs lmarkIR = {l_id, l_xf, l_yf};

      // check if map landmark is in particle sensor range
      if ((fabs(p_x - l_xf) <= sensor_range) &&
          (fabs(p_y - l_yf) <= sensor_range)) {
        landmarksInRange.push_back(lmarkIR);
      }
    }

    // perform transformed observations - in range map landmarks association
    dataAssociation(landmarksInRange, transfOBS);

    // reset weight to 1
    particles[i].weight = 1.0;

    // go through all transformed observations to calculate weights
    for (unsigned j = 0; j < transfOBS.size(); j++) {

      // observed and predicted coordinates
      double tobs_x, tobs_y, lmark_x, lmark_y, snoise_x, snoise_y, gdistmult;

      tobs_x = transfOBS[j].x;
      tobs_y = transfOBS[j].y;

      // iterate through landmarksInRange to get one with match of id
      for (unsigned int z = 0; z < landmarksInRange.size(); z++) {
        if (landmarksInRange[z].id == transfOBS[j].id) {
          lmark_x = landmarksInRange[z].x;
          lmark_y = landmarksInRange[z].y;
        }
      }

      // calculate Multivariate-Gaussian probability
      snoise_x = std_landmark[0];
      snoise_y = std_landmark[1];
      gdistmult = (1 / (2 * M_PI * snoise_x * snoise_y)) *
                  exp(-(((pow(tobs_x - lmark_x, 2)) / (2 * pow(snoise_x, 2))) +
                        (pow(tobs_y - lmark_y, 2) / (2 * pow(snoise_y, 2)))));

      particles[i].weight *= gdistmult;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight. NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // get all weights from particles
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // distribution code from
  // http://www.cplusplus.com/reference/random/discrete_distribution/ and QA
  // video
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> resampledP;

  for (int i = 0; i < num_particles; i++) {
    resampledP.push_back(particles[distribution(gen)]);
  }

  // switch with new particles
  particles = resampledP;
}

Particle ParticleFilter::SetAssociations(Particle particle,
                                         std::vector<int> associations,
                                         std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  // Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
