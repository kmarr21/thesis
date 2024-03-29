// Model-Based model

data {
    // Metadata
    int<lower=1> T;  // Number of trials
    // int N;  // Number of participants 

    int<lower=0, upper=1> Y[T,2]; // choice data for each level (2 = right, 1 = left)
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
    real eta_pr; 
}
transformed parameters {
    //
    real<lower=0, upper=20> beta1;
    real<lower=0, upper=20> beta2;
    real<lower=0, upper=1> eta;

    beta1 = Phi_approx(beta1_pr) * 20;
    beta2 = Phi_approx(beta2_pr) * 20;
    eta = Phi_approx(eta_pr);
}
model {

    // Initialize values
    vector[6] Q = rep_vector(0.5,6); // Model-based Q-values
    // (level 1: 1(L), 2(R), level 2: 3(LL), 4(LR), 5(RL), 6(RR))

    vector[T] deV1 = rep_vector(0, T); // difference in value, level 1
    vector[T] deV2 = rep_vector(0, T); // difference in value, level 2

    // Priors
    beta1_pr ~ normal(0,1);
    beta2_pr ~ normal(0,1);
    eta_pr ~ normal(0,1);

    // Iterate through trials
    for (i in 1:T) {
        // Compute Q values
        Q[1] += 0.7*fmax(Q[3], Q[4]) + 0.3*fmax(Q[5], Q[6]);
        Q[2] += 0.3*fmax(Q[3], Q[4]) + 0.7*fmax(Q[5], Q[6]);

        // Choice likelihood for level 1
        deV1[i] = Q[2] - Q[1];

        // Observe level 2 
        // Choice likelihood for level 2
        // Left: O[i] = 0, Right: O[i] = 1
        deV2[i] = Q[4 + (O[i]*2)] - Q[3 + (O[i]*2)];

        // Update Q value of chosen option
        // Note: using assumption O data comes in as 0 (left) or 1 (right) from stage 1 choice
        Q[3 + (O[i]*2) + Y[i,2]] += eta * (reward[i] - Q[3 + (O[i]*2) + Y[i,2]]);
    }
    // Assign likelihoods
    Y[:,1] ~ bernoulli_logit( beta1 * deV1);
    Y[:,2] ~ bernoulli_logit( beta2 * deV2 );
}
