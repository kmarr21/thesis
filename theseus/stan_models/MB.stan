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

    real  beta_1_pr; 
    real  beta_2_pr;
    real  eta_2_pr;

}
transformed parameters {

    real  beta_1 = beta_1_pr * 10;
    real  beta_2 = beta_2_pr * 10;
    real  eta_2  = Phi_approx(eta_2_pr);

}
model {    

    // Priors
    beta_1_pr ~ std_normal();
    beta_2_pr ~ std_normal();
    eta_2_pr  ~ std_normal();
    
    // Likelihood block
    {
    
    // Preallocate space
    vector[T]    V1;
    vector[T]    V2;
    vector[2]    MB = rep_vector(0.5, 2);
    matrix[2,2]  Q  = rep_matrix(0.5, 2, 2);
    
    // Main loop
    for (i in 1:T) {
    
        // Stage 1 choice
        V1[i] = MB[2] - MB[1];
        
        // Stage 2 choice
        V2[i] = Q[S2[i]+1, 2] - Q[S2[i]+1, 1];
        
        // Update Q-values
        Q[S2[i]+1, Y2[i]+1] += eta_2 * ( R[i] - Q[S2[i]+1, Y2[i]+1] );
        
        // Update MB values
        MB[1] = 0.7 * max(Q[1]) + 0.3 * max(Q[2]);
        MB[2] = 0.3 * max(Q[1]) + 0.7 * max(Q[2]);
    
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
    matrix[2,2]  Q  = rep_matrix(0.5, 2, 2);
    
    // Main loop
    for (i in 1:T) {
    
        // Stage 1 choice
        Y1_pd += exp( bernoulli_logit_lpmf( Y1[i] | beta_1 * (MB[2] - MB[1]) ) );
        
        // Stage 2 choice
        Y2_pd += exp( bernoulli_logit_lpmf( Y2[i] | beta_2 * (Q[S2[i]+1, 2] - Q[S2[i]+1, 1]) ) );
        
        // Update Q-values
        Q[S2[i]+1, Y2[i]+1] += eta_2 * ( R[i] - Q[S2[i]+1, Y2[i]+1] );
        
        // Update MB values
        MB[1] = 0.7 * max(Q[1]) + 0.3 * max(Q[2]);
        MB[2] = 0.3 * max(Q[1]) + 0.7 * max(Q[2]);
    
    }
    
    // Normalize
    Y1_pd /= T;
    Y2_pd /= T;
    
    }
    
}
