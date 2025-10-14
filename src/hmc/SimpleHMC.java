package hmc;


import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.inference.Distribution;
import beast.base.inference.Operator;
import beast.base.inference.State;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;

import java.util.Arrays;

public class SimpleHMC extends Operator {
	final public Input<RealParameter> parameterInput = new Input<>("parameter", "parameter on which to operate", Validate.REQUIRED);
	final public Input<Distribution> likelihoodInput = new Input<>("likelihood", "distribution affected by parameter change", Validate.REQUIRED);
	final public Input<State> stateInput = new Input<>("state", "state containing parameter of interest -- used for numerical gradient calculation", Validate.REQUIRED);

    final public Input<Integer> nStepsInput = new Input<>("stepCount","number of leapfrog steps", 10);
    final public Input<Double> stepSizeInput = new Input<>("stepSize", "size of leapfrog steps", Validate.REQUIRED);

	
    private int numLeapfrogSteps;
    private double epsilon;
    private int dimension;
    
    private RealParameter parameter;
    private Distribution likelihood;
    private State state;
    private double [] gradient;
    
    private double lower;
    
	@Override
	public void initAndValidate() {
		parameter = parameterInput.get();
		state = stateInput.get();
		likelihood = likelihoodInput.get();
		dimension = parameter.getDimension();
		epsilon = stepSizeInput.get();
		numLeapfrogSteps = nStepsInput.get();
		
		gradient = new double[dimension];
		
		lower = 0.1;
	}

	@Override
	public double proposal() {

		double [] currentPosition = parameter.getDoubleValues();
        // Step 1: Sample momentum from a standard normal distribution
        double[] currentMomentum = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            currentMomentum[i] = Randomizer.nextGaussian();
        	if (currentMomentum[i] < -1) {
        		currentMomentum[i] = -1;
        	} else if (currentMomentum[i] > 1) {
        		currentMomentum[i] = 1;
        	}
        }

        // Make copies for the proposal
        double[] proposedPosition = Arrays.copyOf(currentPosition, dimension);
        double[] proposedMomentum = Arrays.copyOf(currentMomentum, dimension);

        // Step 2: Simulate dynamics using the leapfrog integrator
        for (int step = 0; step < numLeapfrogSteps; step++) {
        	
            // Half step for momentum
            proposedMomentum = add(proposedMomentum, multiply(getGradient(proposedPosition), 0.5 * epsilon));
            for (int i = 0; i < dimension; i++) {
            	if (proposedMomentum[i] < -1) {
            		proposedMomentum[i] = -1;
            	} else if (proposedMomentum[i] > 1) {
            		proposedMomentum[i] = 1;
            	}
            }
            // Full step for position
            proposedPosition = add(proposedPosition, multiply(proposedMomentum, epsilon));
            // make sure parameter remains within bounds
            for (int i = 0; i < dimension; i++) {
            	if (proposedPosition[i] < lower) {
            		proposedPosition[i] = lower;
            	}
            }
            // Half step for momentum
            proposedMomentum = add(proposedMomentum, multiply(getGradient(proposedPosition), 0.5 * epsilon));
            for (int i = 0; i < dimension; i++) {
            	if (proposedMomentum[i] < -1) {
            		proposedMomentum[i] = -1;
            	} else if (proposedMomentum[i] > 1) {
            		proposedMomentum[i] = 1;
            	}
            }
        }

        // Negate momentum at the end of the trajectory to make the proposal symmetric
        proposedMomentum = multiply(proposedMomentum, -1.0);

        
//        state.store(-1);
        for (int i = 0; i < dimension; i++) {
        	parameter.setValue(i, proposedPosition[i]);
		}

        double kineticEnergyCurrent = 0.5 * dotProduct(currentMomentum, currentMomentum);
		double kineticEnergyProposed = 0.5 * dotProduct(proposedMomentum, proposedMomentum);
		double logHR = kineticEnergyProposed - kineticEnergyCurrent; 
		return logHR;
    }


	final double DELTA = 1e-6;
	
    private double[] getGradient(double[] proposedPosition) {
//    	state.store(-1);
        for (int i = 0; i < dimension; i++) {
        	if (proposedPosition[i] < lower) {
        		proposedPosition[i] = lower;
        	}
        	if (proposedPosition[i] > parameter.getUpper()) {
        		proposedPosition[i] = parameter.getUpper();
        	}
        	parameter.setValue(i, proposedPosition[i]);
		}
        state.storeCalculationNodes();
        state.checkCalculationNodesDirtiness();
        double logP0 = likelihood.calculateLogP();
        state.acceptCalculationNodes();
        
        for (int i = 0; i < dimension; i++) {
//        	state.store(-1);
        	parameter.setValue(i, proposedPosition[i] + DELTA);
            state.storeCalculationNodes();
            state.checkCalculationNodesDirtiness();
            double logP1 = likelihood.calculateLogP();
            state.restore();
            state.restoreCalculationNodes();
        	
            gradient[i] = (logP0 - logP1) / DELTA;
            if (Double.isNaN(gradient[i])) {
            	int h = 3;
            	h++;
            }
		}
        
		return gradient;
	}

	// Helper method for vector addition
    private double[] add(double[] a, double[] b) {
        double[] result = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    // Helper method for scalar multiplication
    private double[] multiply(double[] v, double scalar) {
        double[] result = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            result[i] = v[i] * scalar;
        }
        return result;
    }

    // Helper method for dot product
    private double dotProduct(double[] a, double[] b) {
        double result = 0.0;
        for (int i = 0; i < dimension; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

//    public double[][] sample(int numSamples, double[] initialPosition) {
//        double[][] samples = new double[numSamples][dimension];
//        samples[0] = initialPosition;
//        double[] currentPosition = Arrays.copyOf(initialPosition, dimension);
//
//        for (int i = 1; i < numSamples; i++) {
//            // Step 1: Sample momentum from a standard normal distribution
//            double[] currentMomentum = new double[dimension];
//            for (int j = 0; j < dimension; j++) {
//                currentMomentum[j] = Randomizer.nextGaussian();
//            }
//
//            // Make copies for the proposal
//            double[] proposedPosition = Arrays.copyOf(currentPosition, dimension);
//            double[] proposedMomentum = Arrays.copyOf(currentMomentum, dimension);
//
//            // Step 2: Simulate dynamics using the leapfrog integrator
//            for (int step = 0; step < numLeapfrogSteps; step++) {
//                // Half step for momentum
//                proposedMomentum = add(proposedMomentum, multiply(getGradient(proposedPosition), 0.5 * epsilon));
//                // Full step for position
//                proposedPosition = add(proposedPosition, multiply(proposedMomentum, epsilon));
//                // Half step for momentum
//                proposedMomentum = add(proposedMomentum, multiply(getGradient(proposedPosition), 0.5 * epsilon));
//            }
//
//            // Negate momentum at the end of the trajectory to make the proposal symmetric
//            proposedMomentum = multiply(proposedMomentum, -1.0);
//
//            // Step 3: Metropolis-Hastings acceptance step
//            double currentEnergy = calculateEnergy(currentPosition, currentMomentum);
//            double proposedEnergy = calculateEnergy(proposedPosition, proposedMomentum);
//
//            if (Randomizer.nextDouble() < Math.exp(currentEnergy - proposedEnergy)) {
//                currentPosition = proposedPosition;
//            }
//            samples[i] = currentPosition;
//        }
//        return samples;
//    }

//    private double calculateEnergy(double[] position, double[] momentum) {
//        // Potential Energy: -log(target density)
//        double potentialEnergy = -Math.log(targetDistribution.density(position));
//        // Kinetic Energy: 0.5 * p^T * p (assuming identity mass matrix)
//        double kineticEnergy = 0.5 * dotProduct(momentum, momentum);
//        return potentialEnergy + kineticEnergy;
//    }
//
//    private double[] getGradient(double[] position) {
//        // For a multivariate normal distribution with mean mu and covariance Sigma,
//        // the gradient of the log-density is -Sigma^-1 * (x - mu).
//        // For simplicity, we assume a standard multivariate normal (mu=0, Sigma=I).
//        // In this case, the gradient is simply -x.
//        return multiply(position, -1.0);
//    }

//    public static void main(String[] args) {
//        // Define a 2D multivariate normal distribution with mean (0, 0) and identity covariance
//        double[] means = {0.0, 0.0};
//        double[][] covariance = {{1.0, 0.0}, {0.0, 1.0}};
//        MultivariateNormalDistribution target = new MultivariateNormalDistribution(means, covariance);
//
//        int numSamples = 100000;
//        int numLeapfrogSteps = 20;
//        double epsilon = 0.1;
//        double[] initialPosition = {0.0, 0.0};
//
//        SimpleHMC hmc = new SimpleHMC(target, numLeapfrogSteps, epsilon);
//        double[][] samples = hmc.sample(numSamples, initialPosition);
//
//        // Print some of the generated samples
//        for (int i = 0; i < 10; i++) {
//            System.out.println("Sample " + i + ": " + Arrays.toString(samples[i]));
//        }
//        
//        MultivariateSummaryStatistics stats = new MultivariateSummaryStatistics(2, false);
//        for (double [] data : samples) {
//        	stats.addValue(data);
//        }
//        RealMatrix cov = stats.getCovariance();
//        double [] mean = stats.getMean();
//        System.out.println("mean: " + Arrays.toString(mean));
//        System.out.println("cov: " + cov.toString());
//    }

}