package hmc.operator;
/*
 * HamiltonianMonteCarloOperator.java
 *
 * Copyright © 2002-2024 the BEAST Development Team
 * http://beast.community/about
 *
 * This file is part of BEAST.
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership and licensing.
 *
 * BEAST is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 *  BEAST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with BEAST; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301  USA
 *
 */


import java.util.ArrayList;

import java.text.DecimalFormat;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Input.Validate;
import beast.base.inference.Distribution;
import beast.base.inference.Operator;
import beast.base.inference.parameter.RealParameter;
import beast.base.util.Randomizer;
import hmc.datastructures.MultivariateFunction;
import hmc.datastructures.ReadableVector;
import hmc.datastructures.WrappedVector;
import hmc.distribution.MultivariateNormalDistribution;

/**
 * @author Max Tolkoff
 * @author Zhenyu Zhang
 * @author Marc A. Suchard
 */

@Description("Vanilla Hamiltonian Monte Carlo operator")
public class HamiltonianMonteCarloOperator extends Operator {
	final public Input<RealParameter> parameterInput = new Input<>("parameter", "parameter on which to operate", Validate.REQUIRED);
	final public Input<Distribution> likelihoodInput = new Input<>("likelihood", "distribution affected by parameter change", Validate.REQUIRED);

	public Input<Boolean> optimiseInput = new Input<>("optimise", "flag to indicate whether to optimise or not", true);
//    public Input<GradientWrtParameterProvider> gradientProviderInput = new Input<>("gradientProvider", "description here");
    public Input<Transform> transformInput = new Input<>("transform", "description here");
    public Input<RealParameter> maskParameterInput = new Input<>("maskParameter", "description here");
//    public Input<Options> runtimeOptionsInput = new Input<>("runtimeOptions", "description here");
    public Input<MassPreconditioningOptions> massPreconditioningOptionsInput = new Input<>("massPreconditioningOptions", "description here");
    public Input<MassPreconditioner.Type> preconditionerInput = new Input<>("preconditioner", "description here", 
    		MassPreconditioner.Type.FULL, MassPreconditioner.Type.values());
    public Input<MassPreconditionScheduler.Type> preconditionerTypeInput = new Input<>("preconditionertype", "description here", 
    		MassPreconditionScheduler.Type.DEFAULT, MassPreconditionScheduler.Type.values());
    
    private final static String PRECONDITIONING_UPDATE_FREQUENCY = "preconditioningUpdateFrequency";
    final static String PRECONDITIONING_MAX_UPDATE = "preconditioningMaxUpdate";
    final static String PRECONDITIONING_DELAY = "preconditioningDelay";
    private final static String PRECONDITIONING_MEMORY = "preconditioningMemory";
    private final static String PRECONDITIONING_GUESS_INIT_MASS = "guessInitialMass";

	final public Input<Integer> preconditioningUpdateFrequencyInput = new Input<>(PRECONDITIONING_UPDATE_FREQUENCY,"PRECONDITIONING_UPDATE_FREQUENCY", 0);
    final public Input<Integer> preconditioningMaxUpdateInput = new Input<>(PRECONDITIONING_MAX_UPDATE,"PRECONDITIONING_MAX_UPDATE", 0);
    final public Input<Integer> preconditioningDelayInput = new Input<>(PRECONDITIONING_DELAY,"PRECONDITIONING_DELAY", 0);
    final public Input<Integer> preconditioningMemoryInput = new Input<>(PRECONDITIONING_MEMORY,"PRECONDITIONING_MEMORY", 0);
    final public Input<Boolean> guessInitialMassInput = new Input<>(PRECONDITIONING_GUESS_INIT_MASS,"PRECONDITIONING_GUESS_INIT_MASS", false);

    public final static String N_STEPS = "nSteps";
    public final static String STEP_SIZE = "stepSize";
    public final static String GRADIENT_CHECK_COUNT = "gradientCheckCount";
    public final static String GRADIENT_CHECK_TOLERANCE = "gradientCheckTolerance";
    
    private final static String MAX_ITERATIONS = "checkStepSizeMaxIterations";
    private final static String REDUCTION_FACTOR = "checkStepSizeReductionFactor";
    private final static String TARGET_ACCEPTANCE_PROBABILITY = "targetAcceptanceProbability";
    private final static String INSTABILITY_HANDLER = "instabilityHandler";
    
    final public Input<Integer> nStepsInput = new Input<>(N_STEPS,"N_STEPS", 10);
    final public Input<Double> stepSizeInput = new Input<>(STEP_SIZE, "step size", Validate.REQUIRED);
    final public Input<Integer> gradientCheckCountInput = new Input<>(GRADIENT_CHECK_COUNT,"GRADIENT_CHECK_COUNT", 0);
    final public Input<Double> gradientCheckToleranceInput = new Input<>(GRADIENT_CHECK_TOLERANCE,"GRADIENT_CHECK_TOLERANCE", 1E-3);
    final public Input<Integer> maxIterationsInput = new Input<>(MAX_ITERATIONS,"MAX_ITERATIONS", 10);
    final public Input<Double> reductionFactorInput = new Input<>(REDUCTION_FACTOR,"REDUCTION_FACTOR", 0.1);
    final public Input<Double> targetAcceptanceProbabilityInput = new Input<>(TARGET_ACCEPTANCE_PROBABILITY,"TARGET_ACCEPTANCE_PROBABILITY", 0.8); // Stan default
    final public Input<String> instabilityHandlerCaseInput = new Input<>(INSTABILITY_HANDLER,"INSTABILITY_HANDLER", "reject");
    final public Input<RealParameter> maskInput = new Input<>("mask", "mask");
    final public Input<Double> randomStepFractionInput = new Input<>("randomStepFraction", "randomStepFractionInput", 0.0);
    final public Input<Double> eigenLowerBoundInput = new Input<>("eigenLowerBound","eigenLowerBoundLower",1E-2);
    final public Input<Double> eigenUpperBoundInput = new Input<>("eigenUpperBound","eigenUpperBoundUpper",1E2);
    
    		
    // Type from MassPreconditionScheduler.Type
    public enum Type {none, _default};
    public Input<Type> preconditionSchedulerTypeInput = new Input<>("preconditionSchedulerType", "description here", Type._default, Type.values());

    GradientWrtParameterProvider gradientProvider;
    protected double stepSize;
    LeapFrogEngine leapFrogEngine;
    protected RealParameter parameter;
    protected MassPreconditioner preconditioning;
    protected MassPreconditionScheduler preconditionScheduler;
    private Options runtimeOptions;
    protected double[] mask;
    protected Transform transform;
    protected Distribution likelihood;
	
	@Override
	public void initAndValidate() {
		this.likelihood = likelihoodInput.get();
        this.parameter = parameterInput.get();

        
        this.gradientProvider = new GradientWrtParameterProvider.ParameterWrapper(new MultivariateNormalDistribution(new double[] {0}, 1), parameter, likelihood);
		
        this.stepSize = stepSizeInput.get(); // runtimeOptions.initialStepSize;

        MassPreconditioner.Type preconditioningType = preconditionerInput.get(); 
        MassPreconditioningOptions options = massPreconditioningOptionsInput.get();
		int preconditioningUpdateFrequency = preconditioningUpdateFrequencyInput.get();
        int preconditioningMaxUpdate = preconditioningMaxUpdateInput.get();
        int preconditioningDelay = preconditioningDelayInput.get();
        int preconditioningMemory = preconditioningMemoryInput.get();
        boolean guessInitialMass = guessInitialMassInput.get();

        
        GradientWrtParameterProvider derivative = this.gradientProvider;

        RealParameter eigenLowerBound = new RealParameter(eigenLowerBoundInput.get()+"");  
        RealParameter eigenUpperBound = new RealParameter(eigenUpperBoundInput.get()+"");
        
        options = new MassPreconditioningOptions.Default(
        			preconditioningUpdateFrequency, 
        			preconditioningMaxUpdate, 
        			preconditioningDelay, 
        			preconditioningMemory, 
        			guessInitialMass, 
        			eigenLowerBound, 
        			eigenUpperBound);
        this.preconditioning = preconditioningType.factory(gradientProvider, transform, options);

        MassPreconditionScheduler.Type preconditionSchedulerType = preconditionerTypeInput.get();

        int nSteps = nStepsInput.get();
        double stepSize = stepSizeInput.get();

        PreconditionHandler preconditionHandler = 
        		new PreconditionHandler(preconditioning,
        				options,
        				preconditionSchedulerType);

        double randomStepFraction = randomStepFractionInput.get();
        if (randomStepFraction > 1) {
            throw new IllegalArgumentException("Random step count fraction must be < 1.0");
        }

        if (parameter == null) {
            parameter = derivative.getParameter();
        }


        boolean dimensionMismatch = derivative.getDimension() != parameter.getDimension();
        if (transform instanceof Transform.MultivariableTransform) {
            dimensionMismatch = ((Transform.MultivariableTransform) transform).getDimension() != parameter.getDimension();
        }

        if (dimensionMismatch) {
            throw new IllegalArgumentException("Gradient (" + derivative.getDimension() +
                    ") must be the same dimensions as the parameter (" + parameter.getDimension() + ")");
        }

        if (preconditionHandler.getMassPreconditioner().getDimension() != parameter.getDimension()) {
            throw new IllegalArgumentException("preconditioner dimension mismatch." + preconditionHandler.getMassPreconditioner().getDimension() + " != " + derivative.getDimension());
        }

        RealParameter mask = null;
        if (maskInput.get() != null) {
            mask = maskInput.get();

            dimensionMismatch = mask.getDimension() != derivative.getDimension();

            if (transform instanceof Transform.MultivariableTransform) {
                dimensionMismatch = ((Transform.MultivariableTransform) transform).getDimension() != mask.getDimension();
            }

            if (dimensionMismatch) {
                throw new IllegalArgumentException("Mask (" + mask.getDimension()
                        + ") must be the same dimension as the gradient (" + derivative.getDimension() + ")");
            }
        }

        int gradientCheckCount = gradientCheckCountInput.get();
        double gradientCheckTolerance = gradientCheckToleranceInput.get();
        int maxIterations = maxIterationsInput.get();
        double reductionFactor = reductionFactorInput.get();
        double targetAcceptanceProbability = targetAcceptanceProbabilityInput.get();
        String instabilityHandlerCase = instabilityHandlerCaseInput.get();

        HamiltonianMonteCarloOperator.InstabilityHandler instabilityHandler = HamiltonianMonteCarloOperator.InstabilityHandler.factory(instabilityHandlerCase);

        this.runtimeOptions = new Options(
                stepSize, nSteps, randomStepFraction,
                preconditionHandler.getOptions(),
                gradientCheckCount, gradientCheckTolerance,
                maxIterations, reductionFactor,
                targetAcceptanceProbability,
                instabilityHandler);
        
        this.preconditionScheduler = preconditionSchedulerType.factory(runtimeOptions, (Operator) this);
        this.mask = buildMask(maskParameterInput.get());
        this.transform = transformInput.get();

        this.leapFrogEngine = constructLeapFrogEngine(transform);
	}


//    public HamiltonianMonteCarloOperator(AdaptationMode mode, double weight,
//                                         GradientWrtParameterProvider gradientProvider,
//                                         RealParameter parameter, Transform transform, RealParameter maskParameter,
//                                         Options runtimeOptions,
//                                         MassPreconditioner.Type preconditioningType) {
//
//        super(mode, runtimeOptions.targetAcceptanceProbability);
//
//        setWeight(weight);
//
//        this.gradientProvider = gradientProvider;
//        this.runtimeOptions = runtimeOptions;
//        this.stepSize = runtimeOptions.initialStepSize;
//        this.preconditioning = preconditioningType.factory(gradientProvider, transform, runtimeOptions);
//        this.parameter = parameter;
//        this.mask = buildMask(maskParameter);
//        this.transform = transform;
//
//        this.leapFrogEngine = constructLeapFrogEngine(transform);
//    }
//public HMCOperator(AdaptationMode mode, double weight,
//                                     GradientWrtParameterProvider gradientProvider,
//                                     RealParameter parameter, Transform transform, RealParameter maskParameter,
//                                     Options runtimeOptions,
//                                     MassPreconditioner preconditioner) {
//    this(mode, weight, gradientProvider, parameter, transform, maskParameter, runtimeOptions,
//            preconditioner, MassPreconditionScheduler.Type.DEFAULT);
//}
//
//    public HMCOperator(AdaptationMode mode, double weight,
//                                         GradientWrtParameterProvider gradientProvider,
//                                         RealParameter parameter, Transform transform, RealParameter maskParameter,
//                                         Options runtimeOptions,
//                                         MassPreconditioner preconditioner,
//                                         MassPreconditionScheduler.Type preconditionSchedulerType) {
//
//        super(mode, runtimeOptions.targetAcceptanceProbability);
//
//        setWeight(weight);
//
//        this.gradientProvider = gradientProvider;
//        this.runtimeOptions = runtimeOptions;
//        this.stepSize = runtimeOptions.initialStepSize;
//        this.preconditioning = preconditioner;
//        this.preconditionScheduler = preconditionSchedulerType.factory(runtimeOptions, (AdaptableMCMCOperator) this);
//        this.parameter = parameter;
//        this.mask = buildMask(maskParameter);
//        this.transform = transform;
//
//        this.leapFrogEngine = constructLeapFrogEngine(transform);
//    }

    protected LeapFrogEngine constructLeapFrogEngine(Transform transform) {
        return (transform != null ?
                new LeapFrogEngine.WithTransform(parameter, transform,
                        getDefaultInstabilityHandler(), preconditioning, mask) :
                new LeapFrogEngine.Default(parameter,
                        getDefaultInstabilityHandler(), preconditioning, mask));
    }


    protected double[] buildMask(RealParameter maskParameter) {

        if (maskParameter == null) return null;

        double[] mask = new double[maskParameter.getDimension()];

        for (int i = 0; i < mask.length; ++i) {
            mask[i] = (maskParameter.getValue(i) == 0.0) ? 0.0 : 1.0;
        }

        return mask;
    }

    @Override
    public double proposal() {
    	return doOperation(likelihood);
    }

    public double doOperation(Distribution joint) {

        if (shouldCheckStepSize()) {
            checkStepSize();
        }

        if (shouldCheckGradient()) {
            checkGradient(joint);
        }

        if (preconditionScheduler.shouldUpdatePreconditioning()) {
            updatePreconditioning();
        }

        try {
            return leapFrog();
        } catch (NumericInstabilityException e) {
            return Double.NEGATIVE_INFINITY;
        } catch (ArithmeticException e) {
            if (REJECT_ARITHMETIC_EXCEPTION) {
                return Double.NEGATIVE_INFINITY;
            } else {
                throw e;
            }
        }
    }

    private void updatePreconditioning() {

        double[] lastGradient = leapFrogEngine.getLastGradient();
        double[] lastPosition = leapFrogEngine.getLastPosition();
        double[] currentPosition = leapFrogEngine.getInitialPosition();
        if (preconditionScheduler.shouldStoreSecant(lastGradient, lastPosition)) {
            preconditioning.storeSecant(new WrappedVector.Raw(lastGradient), new WrappedVector.Raw(currentPosition));
        }
        preconditioning.updateMass();
    }

    private static final boolean REJECT_ARITHMETIC_EXCEPTION = true;

//    @Override
    public void setPathParameter(double beta) {
        if (gradientProvider instanceof PathGradient) {
            ((PathGradient) gradientProvider).setPathParameter(beta);
        }
    }

    private boolean shouldCheckStepSize() {
        return getCount() < 1 && optimiseInput.get(); // getMode() == AdaptationMode.ADAPTATION_ON;
    }

    private void checkStepSize() {

        double[] initialPosition = parameter.getDoubleValues();

        int iterations = 0;
        boolean acceptableSize = false;

        while (!acceptableSize && iterations < runtimeOptions.checkStepSizeMaxIterations) {

            try {
                leapFrog();
                double logLikelihood = gradientProvider.getLikelihood().calculateLogP();

                if (!Double.isNaN(logLikelihood) && !Double.isInfinite(logLikelihood)) {
                    acceptableSize = true;
                }
            } catch (Exception exception) {
                // Do nothing
            }

            if (!acceptableSize) {
                stepSize *= runtimeOptions.checkStepSizeReductionFactor;
            }
            
            //ReadableVector.Utils.setParameter(initialPosition, parameter);  // Restore initial position
            for (int i = 0; i < initialPosition.length; i++) {
            	parameter.setValue(i, initialPosition[i]);
            }
            ++iterations;
        }

        if (!acceptableSize && iterations < runtimeOptions.checkStepSizeMaxIterations) {
            throw new RuntimeException("Unable to find acceptable initial HMC step-size");
        }
    }

    boolean shouldCheckGradient() {
        return getCount() < runtimeOptions.gradientCheckCount;
    }
    
    int getCount() {
    	return m_nNrAccepted + m_nNrRejected;
    }

    void checkGradient(final Distribution joint) {

        if (parameter.getDimension() != gradientProvider.getDimension()) {
            throw new RuntimeException("Unequal dimensions");
        }

        MultivariateFunction numeric = new MultivariateFunction() {

            @Override
            public double evaluate(double[] argument) {

                if (transform == null) {

                    ReadableVector.Utils.setParameter(argument, parameter);
                    return joint.calculateLogP();
                } else {

                    double[] untransformedValue = transform.inverse(argument, 0, argument.length);
                    ReadableVector.Utils.setParameter(untransformedValue, parameter);
                    return joint.calculateLogP() - transform.logJacobian(untransformedValue, 0, untransformedValue.length);
                }
            }

            @Override
            public int getNumArguments() {
                return parameter.getDimension();
            }

            @Override
            public double getLowerBound(int n) {
                return parameter.getLower();
            }

            @Override
            public double getUpperBound(int n) {
                return parameter.getUpper();
            }
        };

        double[] analyticalGradientOriginal = gradientProvider.getGradientLogDensity();
        double[] restoredParameterValue = parameter.getDoubleValues();

        if (transform == null) {

            double[] numericGradientOriginal = NumericalDerivative.gradient(numeric, parameter.getDoubleValues());

            if (!isClose(analyticalGradientOriginal, numericGradientOriginal, runtimeOptions.gradientCheckTolerance)) {

                String sb = "Gradients do not match:\n" +
                        "\tAnalytic: " + new WrappedVector.Raw(analyticalGradientOriginal) + "\n" +
                        "\tNumeric : " + new WrappedVector.Raw(numericGradientOriginal) + "\n" +
                        gradientMismatchInformation(analyticalGradientOriginal, numericGradientOriginal);
                throw new RuntimeException(sb);
            }

        } else {

            double[] transformedParameter = transform.transform(parameter.getDoubleValues(), 0,
                    parameter.getDoubleValues().length);
            double[] numericGradientTransformed = NumericalDerivative.gradient(numeric, transformedParameter);

            double[] analyticalGradientTransformed = transform.updateGradientLogDensity(analyticalGradientOriginal,
                    parameter.getDoubleValues(), 0, parameter.getDoubleValues().length);

            if (!isClose(analyticalGradientTransformed, numericGradientTransformed, runtimeOptions.gradientCheckTolerance)) {
                String sb = "Transformed Gradients do not match:\n" +
                        "\tAnalytic: " + new WrappedVector.Raw(analyticalGradientTransformed) + "\n" +
                        "\tNumeric : " + new WrappedVector.Raw(numericGradientTransformed) + "\n" +
                        "\tParameter : " + new WrappedVector.Raw(parameter.getDoubleValues()) + "\n" +
                        "\tTransformed RealParameter : " + new WrappedVector.Raw(transformedParameter) + "\n" +
                        gradientMismatchInformation(analyticalGradientTransformed, numericGradientTransformed);
                throw new RuntimeException(sb);
            }
        }

        ReadableVector.Utils.setParameter(restoredParameterValue, parameter);
    }

    private String gradientMismatchInformation(double[] analyticGradient, double[] numericGradient) {
        int n = analyticGradient.length;
        double maxDiff = 0;
        int maxInd = -1;
        double meanDiff = 0;
        ArrayList<Integer> overIndices = new ArrayList<>();
        double[] absDiffs = new double[n];

        for (int i = 0; i < n; i++) {
            double absDiff = Math.abs(analyticGradient[i] - numericGradient[i]);
            absDiffs[i] = absDiff;
            meanDiff += absDiff;
            if (absDiff > runtimeOptions.gradientCheckTolerance) {
                overIndices.add(i);
            }
            if (absDiff > maxDiff) {
                maxDiff = absDiff;
                maxInd = i;
            }
        }


        meanDiff /= n;

        StringBuilder sb = new StringBuilder();
        sb.append("\tMaximum absolute difference: " + maxDiff + " (at index " + (maxInd) + ")\n");
        sb.append("\tAverage absolute difference: " + meanDiff + "\n");
        sb.append("\tList of all values exceeding the tolerance:\n");
        sb.append("\t\tindex    analytic    numeric    absolute difference\n");

        int ind = 0;
        String spacer = "    ";
        for (int i : overIndices) {

            sb.append("\t\t" + overIndices.get(ind) + spacer + analyticGradient[i] + spacer + numericGradient[i] +
                    spacer + absDiffs[i] + "\n");
            ind++;
        }

        return sb.toString();
    }

    static double[] mask(double[] vector, double[] mask) {

        assert (mask == null || mask.length == vector.length);

        if (mask != null) {
            for (int i = 0; i < vector.length; ++i) {
                if (mask[i] == 0.0) {
                    vector[i] = 0.0;
                }
            }
        }

        return vector;
    }

    static WrappedVector mask(WrappedVector vector, double[] mask) {

        assert (mask == null || mask.length == vector.getDim());

        if (mask != null) {
            for (int i = 0; i < vector.getDim(); ++i) {
                if (mask[i] == 0.0) {
                    vector.set(i, 0.0);
                }
            }
        }

        return vector;
    }

    private static final boolean DEBUG = true;

    public static class Options implements MassPreconditioningOptions {

        final double initialStepSize;
        final int nSteps;
        final double randomStepCountFraction;
        final int gradientCheckCount;
        final MassPreconditioningOptions preconditioningOptions;
        final double gradientCheckTolerance;
        final int checkStepSizeMaxIterations;
        final double checkStepSizeReductionFactor;
        final double targetAcceptanceProbability;
        final InstabilityHandler instabilityHandler;

        public Options(double initialStepSize, int nSteps, double randomStepCountFraction,
                       MassPreconditioningOptions preconditioningOptions,
                       int gradientCheckCount, double gradientCheckTolerance,
                       int checkStepSizeMaxIterations, double checkStepSizeReductionFactor,
                       double targetAcceptanceProbability, InstabilityHandler instabilityHandler) {
            this.initialStepSize = initialStepSize;
            this.nSteps = nSteps;
            this.randomStepCountFraction = randomStepCountFraction;
            this.gradientCheckCount = gradientCheckCount;
            this.gradientCheckTolerance = gradientCheckTolerance;
            this.checkStepSizeMaxIterations = checkStepSizeMaxIterations;
            this.checkStepSizeReductionFactor = checkStepSizeReductionFactor;
            this.targetAcceptanceProbability = targetAcceptanceProbability;
            this.instabilityHandler = instabilityHandler;
            this.preconditioningOptions = preconditioningOptions;
        }

        @Override
        public int preconditioningUpdateFrequency() {
            return preconditioningOptions.preconditioningUpdateFrequency();
        }

        @Override
        public int preconditioningDelay() {
            return preconditioningOptions.preconditioningDelay();
        }

        @Override
        public int preconditioningMaxUpdate() {
            return preconditioningOptions.preconditioningMaxUpdate();
        }

        @Override
        public int preconditioningMemory() {
            return preconditioningOptions.preconditioningMemory();
        }

        @Override
        public RealParameter preconditioningEigenLowerBound() {
            throw new RuntimeException("Not yet implemented.");
        }

        @Override
        public RealParameter preconditioningEigenUpperBound() {
            throw new RuntimeException("Not yet implemented.");
        }
    }

    public static class NumericInstabilityException extends Exception {
    }

    private int getNumberOfSteps() {
        int count = runtimeOptions.nSteps;
        if (runtimeOptions.randomStepCountFraction > 0.0) {
            double draw = count * (1.0 + runtimeOptions.randomStepCountFraction * (Randomizer.nextDouble() - 0.5));
            count = Math.max(1, (int) draw);
        }
        return count;
    }

    public double getKineticEnergy(ReadableVector momentum) {

        final int dim = momentum.getDim();

        double energy = 0.0;
        for (int i = 0; i < dim; i++) {
            energy += momentum.get(i) * preconditioning.getVelocity(i, momentum);
        }
        return energy / 2.0;
    }

    private double leapFrog() throws NumericInstabilityException {

        if (DEBUG) {
            System.err.println("HMC step size: " + stepSize);
        }

        final WrappedVector momentum = mask(preconditioning.drawInitialMomentum(), mask);
        return leapFrogGivenMomentum(momentum);
    }

    protected double leapFrogGivenMomentum(WrappedVector momentum) throws NumericInstabilityException {
        leapFrogEngine.updateMask();
        final double[] position = leapFrogEngine.getInitialPosition();
        leapFrogEngine.projectMomentum(momentum.getBuffer(), position); //if momentum restricted to subspace

        final double prop = getKineticEnergy(momentum) +
                leapFrogEngine.getParameterLogJacobian();

        leapFrogEngine.updateMomentum(position, momentum.getBuffer(),
                mask(gradientProvider.getGradientLogDensity(), mask), stepSize / 2);


        int nStepsThisLeap = getNumberOfSteps();

        for (int i = 0; i < nStepsThisLeap; i++) { // Leap-frog

            try {
                leapFrogEngine.updatePosition(position, momentum, stepSize);
            } catch (ArithmeticException e) {
                throw new NumericInstabilityException();
            }

            if (i < (nStepsThisLeap - 1)) {

                try {
                    leapFrogEngine.updateMomentum(position, momentum.getBuffer(),
                            mask(gradientProvider.getGradientLogDensity(), mask), stepSize);
                } catch (ArithmeticException e) {
                    throw new NumericInstabilityException();
                }
            }
        }

        leapFrogEngine.updateMomentum(position, momentum.getBuffer(),
                mask(gradientProvider.getGradientLogDensity(), mask), stepSize / 2);

        final double res = getKineticEnergy(momentum) +
                leapFrogEngine.getParameterLogJacobian();

        return prop - res; //hasting ratio
    }


    @Override
    public double getCoercableParameterValue() {
        return Math.log(stepSize);
    }

    @Override
    public void setCoercableParameterValue(double value) {
        if (DEBUG) {
            System.err.println("Setting adaptable parameter: " + getCoercableParameterValue() + " -> " + value);
        }
        stepSize = Math.exp(value);
    }

    /**
     * automatic parameter tuning *
     */
    @Override
    public void optimize(final double logAlpha) {
        if (optimiseInput.get()) {
            double delta = calcDelta(logAlpha);
            delta += Math.log(1.0 / stepSize - 1.0);
            setCoercableParameterValue(1.0 / (Math.exp(delta) + 1.0));
        }
    }

    @Override
    public String getPerformanceSuggestion() {
        final double prob = m_nNrAccepted / (m_nNrAccepted + m_nNrRejected + 0.0);
        final double targetProb = getTargetAcceptanceProbability();

        double ratio = prob / targetProb;
        if (ratio > 2.0) ratio = 2.0;
        if (ratio < 0.5) ratio = 0.5;

        // new scale factor
        final double sf = Math.pow(stepSize, ratio);

        final DecimalFormat formatter = new DecimalFormat("#.###");
        if (prob < 0.10) {
            return "Try setting scaleFactor to about " + formatter.format(sf);
        } else if (prob > 0.40) {
            return "Try setting scaleFactor to about " + formatter.format(sf);
        } else return "";
    }

    //    @Override
//    public double getRawParameter() {
//        return stepSize;
//    }

    public enum InstabilityHandler {

        REJECT("reject") {
            @Override
            void checkValue(double x) throws NumericInstabilityException {
                if (Double.isNaN(x)) throw new NumericInstabilityException();
            }

            @Override
            void checkPosition(Transform transform, double[] unTransformedPosition) throws NumericInstabilityException {
                if (!transform.isInInteriorDomain(unTransformedPosition, 0, unTransformedPosition.length)) {
                    throw new NumericInstabilityException();
                }
            }

//            @Override
//            void checkEqual(double x, double y, double eps) throws NumericInstabilityException {
//                if (Math.abs(x - y) > eps) {
//                    throw new NumericInstabilityException();
//                }
//            }

            @Override
            boolean checkPositionTransform() {
                return true;
            }
        },

        DEBUG("debug") {
            @Override
            void checkValue(double x) throws NumericInstabilityException {
                if (Double.isNaN(x)) {
                    System.err.println("Numerical instability in HMC momentum; throwing exception");
                    throw new NumericInstabilityException();
                }
            }

            @Override
            void checkPosition(Transform transform, double[] unTransformedPosition) throws NumericInstabilityException {
                if (!transform.isInInteriorDomain(unTransformedPosition, 0, unTransformedPosition.length)) {
                    System.err.println("Numerical instability in HMC momentum; throwing exception");
                    throw new NumericInstabilityException();
                }
            }

//            @Override
//            void checkEqual(double x, double y, double eps) throws NumericInstabilityException {
//                if (Math.abs(x - y) > eps) {
//                    System.err.println("Numerical instability in HMC momentum; throwing exception");
//                    throw new NumericInstabilityException();
//                }
//            }

            @Override
            boolean checkPositionTransform() {
                return true;
            }
        },

        IGNORE("ignore") {
            @Override
            void checkValue(double x) {
                // Do nothing
            }

            @Override
            void checkPosition(Transform transform, double[] unTransformedPosition) throws NumericInstabilityException {
                // Do nothing
            }

//            @Override
//            void checkEqual(double x, double y, double eps) {
//                // Do nothing
//            }

            @Override
            boolean checkPositionTransform() {
                return false;
            }
        };

        private final String name;

        InstabilityHandler(String name) {
            this.name = name;
        }

        public static InstabilityHandler factory(String match) {
            for (InstabilityHandler type : InstabilityHandler.values()) {
                if (match.equalsIgnoreCase(type.name)) {
                    return type;
                }
            }
            return null;
        }

        abstract void checkValue(double x) throws NumericInstabilityException;

        //        abstract void checkEqual(double x, double y, double eps) throws NumericInstabilityException;
        abstract void checkPosition(Transform transform, double[] unTransformedPosition) throws NumericInstabilityException;

        abstract boolean checkPositionTransform();
    }

    protected InstabilityHandler getDefaultInstabilityHandler() {
        if (DEBUG) {
            return InstabilityHandler.DEBUG;
        } else {
            return runtimeOptions.instabilityHandler;
        }
    }

    //@Override
    public String getAdaptableParameterName() {
        return "stepSize";
    }

    interface LeapFrogEngine {

        double[] getInitialPosition();

        double getParameterLogJacobian();

        void updateMomentum(final double[] position,
                            final double[] momentum,
                            final double[] gradient,
                            final double functionalStepSize) throws NumericInstabilityException;

        void updatePosition(final double[] position,
                            final WrappedVector momentum,
                            final double functionalStepSize) throws NumericInstabilityException;

        void setParameter(double[] position);

        double[] getLastGradient();

        double[] getLastPosition();

        void projectMomentum(double[] momentum, double[] position);

        void updateMask();

        class Default implements LeapFrogEngine {

            final protected RealParameter parameter;
            final InstabilityHandler instabilityHandler;
            final private MassPreconditioner preconditioning;

            final double[] mask;

            double[] lastGradient;
            double[] lastPosition;

            Default(RealParameter parameter, InstabilityHandler instabilityHandler,
                    MassPreconditioner preconditioning,
                    double[] mask) {
                this.parameter = parameter;
                this.instabilityHandler = instabilityHandler;
                this.preconditioning = preconditioning;
                this.mask = mask;
            }

            @Override
            public double[] getInitialPosition() {
                return parameter.getDoubleValues();
            }

            @Override
            public double getParameterLogJacobian() {
                return 0;
            }

            @Override
            public double[] getLastGradient() {
                return lastGradient;
            }

            @Override
            public double[] getLastPosition() {
                return lastPosition;
            }

            @Override
            public void projectMomentum(double[] momentum, double[] position) {
                // do nothing
            }

            @Override
            public void updateMask() {
                // do nothing
            }

            @Override
            public void updateMomentum(double[] position, double[] momentum, double[] gradient,
                                       double functionalStepSize) throws NumericInstabilityException {

                final int dim = momentum.length;
                for (int i = 0; i < dim; ++i) {
                    momentum[i] += functionalStepSize * gradient[i];
                    instabilityHandler.checkValue(momentum[i]);
                }

                lastGradient = gradient;
                lastPosition = position;
            }

            @Override
            public void updatePosition(double[] position, WrappedVector momentum,
                                       double functionalStepSize) throws NumericInstabilityException {

                final int dim = momentum.getDim();
                for (int i = 0; i < dim; i++) {
                    position[i] += functionalStepSize * preconditioning.getVelocity(i, momentum);
                    instabilityHandler.checkValue(position[i]);
                }

                setParameter(position);
            }

            public void setParameter(double[] position) {
                ReadableVector.Utils.setParameter(position, parameter); // May not work with MaskedParameter?
            }
        }

        class WithTransform extends Default {

            final protected Transform transform;
            double[] unTransformedPosition;

            WithTransform(RealParameter parameter, Transform transform,
                          InstabilityHandler instabilityHandler,
                          MassPreconditioner preconditioning,
                          double[] mask) {
                super(parameter, instabilityHandler, preconditioning, mask);
                this.transform = transform;
            }

            @Override
            public double getParameterLogJacobian() {
                return transform.logJacobian(unTransformedPosition, 0, unTransformedPosition.length);
            }

            @Override
            public double[] getInitialPosition() {
                unTransformedPosition = super.getInitialPosition();
                return transform.transform(unTransformedPosition, 0, unTransformedPosition.length);
            }

            @Override
            public void updateMomentum(double[] position, double[] momentum, double[] gradient,
                                       double functionalStepSize) throws NumericInstabilityException {

                gradient = transform.updateGradientLogDensity(gradient, unTransformedPosition,
                        0, unTransformedPosition.length);
                mask(gradient, mask);
                super.updateMomentum(position, momentum, gradient, functionalStepSize);
            }

            @Override
            public void updatePosition(double[] position, WrappedVector momentum,
                                       double functionalStepSize) throws NumericInstabilityException {

                super.updatePosition(position, momentum, functionalStepSize);

                if (instabilityHandler.checkPositionTransform()) {
                    checkPosition(unTransformedPosition);
                }
            }

            @Override
            public void setParameter(double[] position) {
                unTransformedPosition = transform.inverse(position, 0, position.length);
                super.setParameter(unTransformedPosition);
            }

            private void checkPosition(double[] unTransformedPosition) throws NumericInstabilityException {
                instabilityHandler.checkPosition(transform, unTransformedPosition);
            }

//            private void checkPosition(double[] position) throws NumericInstabilityException {
//                double[] newPosition = transform.transform(transform.inverse(position, 0, position.length),
//                        0, position.length);
//                for (int i = 0; i < position.length; i++) {
//                    instabilityHandler.checkEqual(position[i], newPosition[i], EPS);
//                }
//            }
//
//            private double EPS = 10e-10;
        }
    }

//    @Override
    public void reversiblePositionMomentumUpdate(WrappedVector position, WrappedVector momentum,
                                                 WrappedVector gradient, int direction, double time) {

        preconditionScheduler.forceUpdateCount();
        //providerUpdatePreconditioning();

        try {
            leapFrogEngine.updateMomentum(position.getBuffer(), momentum.getBuffer(),
                    mask(gradient.getBuffer(), mask), time * direction / 2);
            leapFrogEngine.updatePosition(position.getBuffer(), momentum, time * direction);
            updateGradient(gradient);
            leapFrogEngine.updateMomentum(position.getBuffer(), momentum.getBuffer(),
                    mask(gradient.getBuffer(), mask), time * direction / 2);
        } catch (NumericInstabilityException e) {
            handleInstability();
        }
    }

//    @Override
    public void providerUpdatePreconditioning() {
        updatePreconditioning();
    }

    public void updateGradient(WrappedVector gradient) {
        double[] buffer = gradientProvider.getGradientLogDensity();
        for (int i = 0; i < buffer.length; i++) {
            gradient.set(i, buffer[i]);
        }
    }

//    @Override
    public double[] getInitialPosition() {

        return leapFrogEngine.getInitialPosition();
    }

//    @Override
    public double getParameterLogJacobian() {
        return leapFrogEngine.getParameterLogJacobian();
    }

//    @Override
    public Transform getTransform() {
        return transform;
    }

//    @Override
    public GradientWrtParameterProvider getGradientProvider() {
        return gradientProvider;
    }

//    @Override
    public void setParameter(double[] position) {
        leapFrogEngine.setParameter(position);
    }

//    @Override
    public WrappedVector drawMomentum() {
        return mask(preconditioning.drawInitialMomentum(), mask);
    }

//    @Override
    public double getJointProbability(WrappedVector momentum) {
        return gradientProvider.getLikelihood().calculateLogP() - getKineticEnergy(momentum) - getParameterLogJacobian();
    }

//    @Override
    public double calculateLogP() {
        return gradientProvider.getLikelihood().calculateLogP();
    }

//    @Override
    public double getStepSize() {
        return stepSize;
    }

    public int getNumGradientEvent(){
        return 0;
    }

//    @Override
    public int getNumBoundaryEvent() {
        return 0;
    }

//    @Override
    public double[] getMask() {
        return mask;
    }

    protected void handleInstability() {
        throw new RuntimeException("Numerical instability; need to handle"); // TODO
    }
    
    
    public static boolean isClose(double[] x, double[] y, double tolerance) {
        if (x.length != y.length) return false;

        for (int i = 0, dim = x.length; i < dim; ++i) {
            if (Double.isNaN(x[i]) || Double.isNaN(y[i])) return false;
            if (Math.abs(x[i] - y[i]) > tolerance) return false;
        }

        return true;
    }

    public static boolean isClose(double x, double y, double tolerance) {
        return Math.abs(x - y) < tolerance;
    }

}