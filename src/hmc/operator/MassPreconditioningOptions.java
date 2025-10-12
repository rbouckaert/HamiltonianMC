/*
 * MassPreconditioningOptions.java
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

package hmc.operator;

import beast.base.inference.parameter.RealParameter;

public interface MassPreconditioningOptions {

    int preconditioningUpdateFrequency();
    int preconditioningDelay();
    int preconditioningMaxUpdate();
    int preconditioningMemory();
    RealParameter preconditioningEigenLowerBound();
    RealParameter preconditioningEigenUpperBound();

    class Default implements MassPreconditioningOptions {
        final int preconditioningUpdateFrequency;
        final int preconditioningMaxUpdate;
        final int preconditioningDelay;
        final int preconditioningMemory;
        final boolean guessInitialMass;
        final RealParameter preconditioningEigenLowerBound;
        final RealParameter preconditioningEigenUpperBound;

        public Default(int preconditioningUpdateFrequency, int preconditioningMaxUpdate,
                       int preconditioningDelay, int preconditioningMemory, boolean guessInitialMass,
                       RealParameter eigenLowerBound, RealParameter eigenUpperBound) {
            this.preconditioningUpdateFrequency = preconditioningUpdateFrequency;
            this.preconditioningMaxUpdate = preconditioningMaxUpdate;
            this.preconditioningDelay = preconditioningDelay;
            this.preconditioningMemory = preconditioningMemory;
            this.guessInitialMass = guessInitialMass;
            this.preconditioningEigenLowerBound = eigenLowerBound;
            this.preconditioningEigenUpperBound = eigenUpperBound;
        }

        @Override
        public int preconditioningUpdateFrequency() {
            return preconditioningUpdateFrequency;
        }

        @Override
        public int preconditioningDelay() {
            return preconditioningDelay;
        }

        @Override
        public int preconditioningMaxUpdate() {
            return preconditioningMaxUpdate;
        }

        @Override
        public int preconditioningMemory() {
            return preconditioningMemory;
        }

        @Override
        public RealParameter preconditioningEigenLowerBound() {
            return preconditioningEigenLowerBound;
        }

        @Override
        public RealParameter preconditioningEigenUpperBound() {
            return preconditioningEigenUpperBound;
        }
    }
}
