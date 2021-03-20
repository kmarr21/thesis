data {

    # Metadata
    int  T;
    
    # Data
    int  Y1[T];        # Stage 1 choice
    int  Y2[T];        # Stage 2 choice
    int  S2[T];        # State 2
    real  R[T];        # Reward

}
parameters {

    # Hyperparameters (group parameters)
    real  w_pr;
    real  p_pr;
    real  lam_pr;
    real  beta_1_pr; 
    real  beta_2_pr;
    real  eta_1_pr;
    real  eta_2_pr;

}
transformed parameters {

    real  w      = Phi_approx(w_pr);
    real  p      = Phi_approx(p_pr);
    real  lam    = Phi_approx(lam_pr);
    real  beta_1 = beta_1_pr * 10;
    real  beta_2 = beta_2_pr * 10;
    real  eta_1  = Phi_approx(eta_1_pr);
    real  eta_2  = Phi_approx(eta_2_pr);

}
model {    

    # Priors
    w_pr      ~ std_normal();
    p_pr      ~ std_normal();
    lam_pr    ~ std_normal();
    beta_1_pr ~ std_normal();
    beta_2_pr ~ std_normal();
    eta_1_pr  ~ std_normal();
    eta_2_pr  ~ std_normal();
    
    # Likelihood block
    {
    
    # Preallocate space
    vector[T]    V1;
    vector[T]    V2;
    vector[2]    MB = rep_vector(0.5, 2);
    vector[2]    MF = rep_vector(0.5, 2);
    vector[2]    HQ = rep_vector(0.5, 2);
    matrix[2,2]  Q  = rep_matrix(0.5, 2, 2);
    int          m;
    
    # Main loop
    for (i in 1:T) {
    
        # Stage 1 choice
        V1[i] = HQ[2] - HQ[1];

        if (i > 1) {
            // m = 1 (if first stage and same as previous trial), otherwise m = 0
            m = 1 ? (Y1[i-1] == 1): -1; // ternary conditional operator
            #m = ((Y1[i-1] == 1) : 1 ? -1);
            V1[i] += p*m;
        }
        
        # Stage 2 choice
        V2[i] = Q[S2[i]+1, 2] - Q[S2[i]+1, 1];

        # Update MF values
        MF[Y1[i]+1] += eta_1 * ( Q[S2[i]+1, Y2[i]+1] - MF[Y1[i]+1] );
        
        # Update Q-values
        Q[S2[i]+1, Y2[i]+1] += eta_2 * ( R[i] - Q[S2[i]+1, Y2[i]+1] );
        
        # Update MB values
        MB[1] = 0.7 * max(Q[1]) + 0.3 * max(Q[2]);
        MB[2] = 0.3 * max(Q[1]) + 0.7 * max(Q[2]);

        # Eligibility update
        MF[Y1[i]+1] += lam * ( R[i] - MF[Y1[i]+1] );
        
        # Update hybrid values
        HQ = w * MB + (1-w) * MF;
    
    }
    
    # Likelihood
    Y1 ~ bernoulli_logit( beta_1 * V1 );
    Y2 ~ bernoulli_logit( beta_2 * V2 );
    
    }
}