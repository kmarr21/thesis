data {

    // Metadata
    int  T;
    
    // Data
    int  Y1[T];        // Stage 1 choice
    int  Y2[T];        // Stage 2 choice
    int  S2[T];        // State 2
    real  R[T];        // Reward

}
parameters {

    real  w_pr;
    real  beta_1_pr; 
    real  beta_2_pr;
    real  eta_1_pr;
    real  eta_2_pr;

}
transformed parameters {

    real  w      = Phi_approx(w_pr);
    real  beta_1 = beta_1_pr * 10;
    real  beta_2 = beta_2_pr * 10;
    real  eta_1  = Phi_approx(eta_1_pr);
    real  eta_2  = Phi_approx(eta_2_pr);

}
model {    

    // Priors
    w_pr      ~ std_normal();
    beta_1_pr ~ std_normal();
    beta_2_pr ~ std_normal();
    eta_1_pr  ~ std_normal();
    eta_2_pr  ~ std_normal();
    
    // Likelihood block
    {
    
    // Preallocate space
    vector[T]    V1;
    vector[T]    V2;
    vector[2]    MB = rep_vector(0.5, 2);
    vector[2]    MF = rep_vector(0.5, 2);
    vector[2]    HQ = rep_vector(0.5, 2);
    matrix[2,2]  Q  = rep_matrix(0.5, 2, 2);
    
    // Main loop
    for (i in 1:T) {
    
        // Stage 1 choice
        V1[i] = HQ[2] - HQ[1];
        
        // Stage 2 choice
        V2[i] = Q[S2[i], 2] - Q[S2[i], 1];
        
        // Update Q-values
        Q[S2[i], Y2[i]+1] += eta_2 * ( R[i] - Q[S2[i], Y2[i]+1] );
        
        // Update MB values
        MB[1] = 0.7 * max(Q[1]) + 0.3 * max(Q[2]);
        MB[2] = 0.3 * max(Q[1]) + 0.7 * max(Q[2]);
        
        // Update MF values
        MF[Y1[i]+1] += eta_1 * ( R[i] - MF[Y1[i]+1] );
        
        // Update hybrid values
        HQ = w * MB + (1-w) * MF;
    
    }
    
    // Likelihood
    Y1 ~ bernoulli_logit( beta_1 * V1 );
    Y2 ~ bernoulli_logit( beta_2 * V2 );
    
    }

}
generated quantities {

    // Choice likelihood
    real  Y1_pd = 0.;
    real  Y2_pd = 0.;
    
    // Likelihood block
    {
    
    // Preallocate space
    vector[2]    MB = rep_vector(0.5, 2);
    vector[2]    MF = rep_vector(0.5, 2);
    vector[2]    HQ = rep_vector(0.5, 2);
    matrix[2,2]  Q  = rep_matrix(0.5, 2, 2);
    
    // Main loop
    for (i in 1:T) {
    
        // Stage 1 choice
        Y1_pd += exp( bernoulli_logit_lpmf( Y1[i] | beta_1 * (HQ[2] - HQ[1]) ) );
        
        // Stage 2 choice
        Y2_pd += exp( bernoulli_logit_lpmf( Y2[i] | beta_2 * (Q[S2[i], 2] - Q[S2[i], 1]) ) );
        
        // Update Q-values
        Q[S2[i], Y2[i]+1] += eta_2 * ( R[i] - Q[S2[i], Y2[i]+1] );
        
        // Update MB values
        MB[1] = 0.7 * max(Q[1]) + 0.3 * max(Q[2]);
        MB[2] = 0.3 * max(Q[1]) + 0.7 * max(Q[2]);
        
        // Update MF values
        MF[Y1[i]+1] += eta_1 * ( R[i] - MF[Y1[i]+1] );
        
        // Update hybrid values
        HQ = w * MB + (1-w) * MF;
    
    }
    
    // Normalize
    Y1_pd /= T;
    Y2_pd /= T;
    
    }
    
}