// DaSilva TLDR Model

data {
    // Metadata
    int T;  // Number of trials
    // int N;  // Number of participants 

    int Y1[T]; // Stage 1 choice
    int Y2[T]; // Stage 2 choice
    int S2[T]; // State 2
    real R[T]; // Reward
}
parameters {
    // Subject-level parameters
    real beta1_pr;
    real beta2_pr;
    real etaC_pr; 
    real etaR_pr;
    real etaF_pr;
    real p_pr;
}
transformed parameters {
    // 
    real<lower=0, upper=20> beta1;
    real<lower=0, upper=20> beta2;
    real<lower=0, upper=1> etaC;
    real<lower=0, upper=1> etaR;
    real<lower=0, upper=1> etaF;
    real p;

    beta1 = Phi_approx(beta1_pr) * 10;
    beta2 = Phi_approx(beta2_pr) * 10;
    etaC = Phi_approx (etaC_pr);
    etaR = Phi_approx (etaR_pr);
    etaF = Phi_approx (etaF_pr);
    p = Phi_approx (p_pr);
}
model {

    // Priors
    beta1_pr ~ std_normal(); 
    beta2_pr ~ std_normal(); 
    etaC_pr ~ std_normal();
    etaR_pr ~ std_normal();
    etaF_pr ~ std_normal();
    p_pr ~ std_normal();

    // Likelihood block 
    {
    // Initialize values
    vector[6] Q = rep_vector(0.5,6); // Model-based Q-values
    // (level 1: 1(L), 2(R), level 2: 3(LL), 4(LR), 5(RL), 6(RR))

    vector[T] deV1 = rep_vector(0, T); // difference in value, level 1
    vector[T] deV2 = rep_vector(0, T); // difference in value, level 2
    vector[T] LR = rep_vector(0, T); // trial learning rate: can be common or rare 
    int m;

    // Iterate through trials
    for (i in 1:T) {

        // Choice likelihood for stage 1
        deV1[i] = Q[2] - Q[1];

        if (i > 1) {
            // m = 1 (if first stage and same as previous trial), otherwise m = 0
            m = 1 ? (Y1[i-1] == 1): -1; // ternary conditional operator
            #m = ((Y1[i-1] == 1) : 1 ? -1);
            deV1[i] += p*m;
        }

        // Observe stage 2 choice
        // Choice likelihood for level 2
        // Left: S2[i] = 0, Right: S2[i] = 1
        deV2[i] = Q[4 + (S2[i]*2)] - Q[3 + (S2[i]*2)];

        // S2[i] = 0, —> 3 or 4
        // S2[i] = 1 —> 5 or 6

        if (S2[i] == Y1[i]) {
            LR[i] = etaC; // common transition
        } else {
            LR[i] = etaR; // rare transition
        }
        // Chosen Action update
        Q[3 + (S2[i]*2) + Y2[i]] += LR[i] * (R[i] - Q[3 + (S2[i]*2) + Y2[i]]);

        // Unchosen Action update
        Q[3 + (S2[i]*2) + (1-Y2[i])] += etaF * (0 - Q[3 + (S2[i]*2) + (1-Y2[i])]);
        Q[1 + (S2[i]*2) + (1-Y2[i])] += etaF * (0 - Q[1 + (S2[i]*2) + (1-Y2[i])]);
        Q[1 + (S2[i]*2) + Y2[i]] += etaF * (0 - Q[1 + (S2[i]*2) + Y2[i]]);

        // Update Q values
        Q[1] = 0.7*fmax(Q[3], Q[4]) + 0.3*fmax(Q[5], Q[6]);
        Q[2] = 0.3*fmax(Q[3], Q[4]) + 0.7*fmax(Q[5], Q[6]);
    }
    
    // Assign likelihoods
    Y1 ~ bernoulli_logit( beta1 * deV1);
    Y2 ~ bernoulli_logit( beta2 * deV2 );

    }
}