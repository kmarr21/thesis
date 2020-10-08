// DaSilva TDLR Model

data {
    // Metadata
    int<lower=1, upper=T> T;  // Number of trials
    // int N;  // Number of participants 

    int<lower=1, upper=2> Y[max(T),2]; // choice data for each level (2 = right, 1 = left)
    int<lower=0, upper=1> O[max(T)]; // outcome data for level 1 choice (1 = right, 0 = left)
    int<lower=0, upper=1> reward[max(T)]; // trial reward
    int<lower=0, upper=1> LR[max(T)]; // trial learning rate -- Need this here?
} 
transformed data {
    // Do I need to vectorize my choices? I am still a bit confused on this?
    // I feel like I need to vectorize Y....
}
parameters {
    // Subject-level parameters
    real beta1_pr;
    real beta2_pr;
    real etaC_pr; 
    real etaR_pr;
}
transformed parameters {
    // I am still confused here?

    real<lower=0, upper=20> beta1;
    real<lower=0, upper=20> beta2;
    real<lower=0, upper=1> etaC;
    real<lower=0, upper=1> etaR;

    beta1 = Phi_approx(beta1_pr) * 20;
    beta2 = Phi_approx(beta2_pr) * 20;
    etaC = Phi_approx(etaC_pr);
    etaR = Phi_approx(etaR_pr);
}
model {

    // Priors
    beta1_pr ~ normal(0,1); 
    beta2_pr ~ normal(0,1); 
    etaC_pr ~ normal(0,1);
    etaR_pr ~ normal(0,1); 

    // Initialize values
    vector[6] Q = rep_vector(0.5,6); // Model-based Q-values
    // (level 1: 1(L), 2(R), level 2: 3(LL), 4(LR), 5(RL), 6(RR))

    vector[T] deV1 = rep_vector(0, T); // difference in value, level 1
    vector[T] deV2 = rep_vector(0, T); // difference in value, level 2

    // Iterate through trials
    for (i in 1:T) {
        // Compute Q values
        Q[1] = 0.7*fmax(Q[3], Q[4]) + 0.3*fmax(Q[5], Q[6]);
        Q[2] = 0.3*fmax(Q[3], Q[4]) + 0.7*fmax(Q[5], Q[6]);

        // Choice likelihood for level 1
        deV1[i] = Q[2] - Q[1];
        Y[i,1] ~ bernoulli_logit( beta1 * deV1);

        // Observe level 2 
        // Choice likelihood for level 2
        if (O[i] == 0) {
            // went Left
            deV2[i] = Q[4] - Q[3]
        } else {
            // went Right
            deV2[i] = Q[6] - Q[5]
        }
        Y[i,2] ~ bernoulli_logit( beta2 * deV2 );

        if (Y[i,1] == O[i]) {
            LR[i] = etaC; // common transition
        } else {
            LR[i] = etaR; // rare transition
        }
        // Update Q value of chosen option
        // Note: using assumption O data comes in as 0 (left) or 1 (right) from stage 1 choice
        Q[(O[i]+ 3) + (2*O[i])] += LR[i] * (reward[i] - Q[(O[i]+3) + (2*O[i])]);
    }

}
