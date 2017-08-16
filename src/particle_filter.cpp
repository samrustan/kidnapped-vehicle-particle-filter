/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 * Modified on: Aug 08, 2017
 * Modified by: Sam Rustan
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// DONE: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;

	normal_distribution<double> n_dist_x(x,std[0]);
	normal_distribution<double> n_dist_y(y,std[1]);
	normal_distribution<double> n_dist_theta(theta,std[2]);

	num_particles = 100;

	Particle particle;

    // generate random values and create particles
	for(int i = 0; i < num_particles; i++) {
		
		particle.id = i;
		particle.x = n_dist_x(gen);
		particle.y = n_dist_y(gen);
		particle.theta = n_dist_theta(gen);
		particle.weight = 1;
		particles.push_back(particle);
		weights.push_back(1);	
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// DONE: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	for(int i = 0; i < particles.size(); i++) {

		double tmp_x;
		double tmp_y;
		double tmp_theta;

		if(fabs(yaw_rate) < 0.0001) {
			tmp_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			tmp_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
			tmp_theta = particles[i].theta;
		}
		else {
			tmp_x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			tmp_y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
			tmp_theta = particles[i].theta + yaw_rate * delta_t;
		}

		normal_distribution<double> n_dist_x(tmp_x,std_pos[0]);
		normal_distribution<double> n_dist_y(tmp_y,std_pos[1]);
		normal_distribution<double> n_dist_theta(tmp_theta,std_pos[2]);		
	
		particles[i].x = n_dist_x(gen);
		particles[i].y = n_dist_y(gen);
		particles[i].theta = n_dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// DONE: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(int i = 0; i < observations.size(); i++) {
 
        // init minimum distance to maximum possible
		double min_distance = numeric_limits<double>::max();

		for(int j = 0; j < predicted.size(); j++) {
 
            // get distance between current/predicted landmarks
			double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

            // find the predicted landmark nearest the current observed landmark
			if(distance < min_distance)	{
				min_distance = distance;
				// set the observation's id to the nearest predicted landmark's id
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	//  DONE: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
        
        // pull values from i iteration for calculation below
	    double p_x = particles[i].x;
	    double p_y = particles[i].y;
	    double p_theta = particles[i].theta;

	    // predicted landmark locations vector
	    vector<LandmarkObs> predictions;

	    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

	    	LandmarkObs predLandmark;

			predLandmark.x = map_landmarks.landmark_list[j].x_f;
			predLandmark.y = map_landmarks.landmark_list[j].y_f;
			predLandmark.id = map_landmarks.landmark_list[j].id_i;

			predictions.push_back(predLandmark);
	    }

	    // transform observations from vehicle to map
		vector<LandmarkObs> trans_observations;
		LandmarkObs obs;

		for(int j = 0; j < observations.size(); j++) {

			LandmarkObs trans_obs;
			obs = observations[j];

			trans_obs.x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + particles[i].x;
	     	trans_obs.y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + particles[i].y;
	      	trans_observations.push_back(trans_obs);
		}

	    // perform dataAssociation for the predictions and transformed observations on current particle
	    dataAssociation(predictions, trans_observations);

	    // re-init weight
	    particles[i].weight = 1.0;

	    // calculate particles final weight using multivariate gaussian probalility
	    for (int j = 0; j < trans_observations.size(); j++) {

			int association_pred = trans_observations[j].id;

			double measured_x = trans_observations[j].x;
			double measured_y = trans_observations[j].y;
            
            // set mu to the values pulled from first (i) iteration x & y
			double mu_x = p_x;
			double mu_y = p_y;

            // get the x,y coordinates of the prediction associated with the current observation
			for (uint k = 0; k < predictions.size(); k++) {
		        if (predictions[k].id == association_pred) {
		          mu_x = predictions[k].x;
		          mu_y = predictions[k].y;
		        }
		    }

			double obs_w = 1/(2*M_PI*std_landmark[0]*std_landmark[1])*exp(-(pow(measured_x-mu_x,2.0)/(2*pow(std_landmark[0],2.0))+pow(measured_y-mu_y,2.0)/(2*pow(std_landmark[1],2.0))));
			// product of the obersvation weight with total observations weight
			particles[i].weight *= obs_w;
			weights[i] = particles[i].weight;
	    }
  	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> resample_particles;

  for(int i = 0; i < num_particles; i++) {
  	resample_particles.push_back(particles[distribution(gen)]);
  }
  particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
  

