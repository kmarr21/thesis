// DaSilva TLDR Model

data {
    // Metadata
    int<lower=1> T;  // Number of trials
    // int N;  // Number of participants 

    int<lower=0, upper=1> Y1[T]; // choice data for stage 1
    int<lower=0, upper=1> Y2[T]; // choice data for stage 2
    int<lower=0, upper=1> O[T]; // outcome data for level 1 choice (1 = right, 0 = left)
    int<lower=0, upper=1> reward[T]; // trial reward
} 
transformed data {
    // 
}
parameters {
    // Subject-level parameters
    real beta1_pr;
    real beta2_pr;
    real etaC_pr; 
    real etaR_pr;
}
transformed parameters {
    // 
    real<lower=0, upper=20> beta1;
    real<lower=0, upper=20> beta2;
    real<lower=0, upper=1> etaC;
    real<lower=0, upper=1> etaR;

    beta1 = Phi_approx(beta1_pr) * 20;
    beta2 = Phi_approx(beta2_pr) * 20;
    etaC = Phi_approx (etaC_pr);
    etaR = Phi_approx (etaR_pr);
}
model {

    // Initialize values
    vector[6] Q = rep_vector(0.5,6); // Model-based Q-values
    // (level 1: 1(L), 2(R), level 2: 3(LL), 4(LR), 5(RL), 6(RR))

    vector[T] deV1 = rep_vector(0, T); // difference in value, level 1
    vector[T] deV2 = rep_vector(0, T); // difference in value, level 2
    vector[T] LR = rep_vector(0, T); // trial learning rate: can be common or rare
    
    // Priors
    beta1_pr ~ normal(0,1); 
    beta2_pr ~ normal(0,1); 
    etaC_pr ~ normal(0,1);
    etaR_pr ~ normal(0,1); 

    // Iterate through trials
    for (i in 1:T) {

        // Choice likelihood for stage 1
        deV1[i] = Q[2] - Q[1];

        // Observe stage 2 choice
        // Choice likelihood for level 2
        // Left: O[i] = 0, Right: O[i] = 1
        deV2[i] = Q[4 + (O[i]*2)] - Q[3 + (O[i]*2)];

        // O[i] = 0, —> 3 or 4 — 4 + (O[i]*2), 3 +
        // O[i] = 1 —> 5 or 6

        if (O[i]) {
            LR[i] = etaC; // common transition
        } else {
            LR[i] = etaR; // rare transition
        }
        // Update Q value of chosen option
        // Note: using assumption O[i] data comes in as 0 (left) or 1 (right) from stage 1 choice
        Q[3 + (O[i]*2) + Y2[i]] += LR[i] * (reward[i] - Q[3 + (O[i]*2) + Y2[i]]);

        // Update Q values
        Q[1] = 0.7*fmax(Q[3], Q[4]) + 0.3*fmax(Q[5], Q[6]);
        Q[2] = 0.3*fmax(Q[3], Q[4]) + 0.7*fmax(Q[5], Q[6]);
    }
    // Assign likelihoods
    Y1 ~ bernoulli_logit( beta1 * deV1);
    Y2 ~ bernoulli_logit( beta2 * deV2 );
}
generated quantities {
    // Choice likelihood
    real Y1_pd = 0.;
    real Y2_pd = 0.;

    // Likelihood block
    {
        // Preallocate space
        vector[6] Q = rep_vector(0.5,6);
        vector[T] LR = rep_vector(0, T);

        // Main loop
        for (i in 1:T) {
            // Stage 1 choice
            Y1_pd += exp( bernoulli_logit_lpmf( Y1[i] | beta1 * (Q[2] - Q[1]) ));

            // Stage 2 choice
            Y2_pd += exp( bernoulli_logit_lpmf( Y2[i] | beta2 * (Q[4 + (O[i]*2)] - Q[3 + (O[i]*2)]) ));

            if (O[i]) {
                LR[i] = etaC; // common transition
            } else {
                LR[i] = etaR; // rare transition
            }
            // Update Q value of chosen option
            Q[3 + (O[i]*2) + Y2[i]] += LR[i] * (reward[i] - Q[3 + (O[i]*2) + Y2[i]]);

            // Update stage 1 Q values
            Q[1] = 0.7*fmax(Q[3], Q[4]) + 0.3*fmax(Q[5], Q[6]);
            Q[2] = 0.3*fmax(Q[3], Q[4]) + 0.7*fmax(Q[5], Q[6]);
        }
        // Normalize
        Y1_pd /= T;
        Y2_pd /= T;
    }
}