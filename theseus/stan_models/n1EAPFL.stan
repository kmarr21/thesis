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

    // Hyperparameters (group parameters)
    real  w_pr;
    real  p_pr;
    real  lam_pr;
    real  beta_1_pr; 
    real  beta_2_pr;
    real  eta_1_pr;
    real  eta_2_pr;
    real  eta_F_pr;

}
transformed parameters {

    real  w      = Phi_approx(w_pr);
    real  p      = Phi_approx(p_pr);
    real  beta_1 = beta_1_pr * 10;
    real  beta_2 = beta_2_pr * 10;
    real  eta_1  = Phi_approx(eta_1_pr);
    real  eta_2  = Phi_approx(eta_2_pr);
    real  eta_F  = Phi_approx(eta_F_pr);
    real  lam    = Phi_approx(lam_pr);

}
model {    

    // Priors
    w_pr      ~ std_normal();
    beta_1_pr ~ std_normal();
    beta_2_pr ~ std_normal();
    eta_1_pr  ~ std_normal();
    eta_2_pr  ~ std_normal();
    eta_F_pr  ~ std_normal();
    p_pr      ~ std_normal();
    lam_pr    ~ std_normal();
    
    // Likelihood block
    {
    
    // Preallocate space
    vector[T]    V1;
    vector[T]    V2;
    vector[2]    Q1 = rep_vector(0.5, 2);
    matrix[2,2]  Q2  = rep_matrix(0.5, 2, 2);
    int          m;
    real         cr;
    real         deltaC;
    real         deltaU;
    
    // Main loop
    for (i in 1:T) {
    
        // Stage 1 choice
        V1[i] = Q1[2] - Q1[1];

        if (i > 1) {
            // m = 1 (if first stage and same as previous trial), otherwise m = 0
            m = 1 ? (Y1[i-1] == 1): -1; // ternary conditional operator
            #m = ((Y1[i-1] == 1) : 1 ? -1);
            V1[i] += p*m;
        }
        
        // Stage 2 choice
        V2[i] = Q2[S2[i]+1, 2] - Q2[S2[i]+1, 1];
        
        // Update stage 2 Q-values
        Q2[S2[i]+1, Y2[i]+1] += eta_2 * ( R[i] - Q2[S2[i]+1, Y2[i]+1] );

        // Forgetting Process
        Q2[S2[i]+1, (1-Y2[i])+1] += eta_F * ( 0 - Q2[S2[i]+1, (1-Y2[i])+1] );
        Q2[(1-S2[i])+1, Y2[i]+1] += eta_F * ( 0 - Q2[(1-S2[i])+1, Y2[i]+1] );
        Q2[(1-S2[i])+1, (1-Y2[i])+1] += eta_F * ( 0 - Q2[(1-S2[i])+1, (1-Y2[i])+1] );
        
        // Transition condition
        if(Y1[i] == S2[i]) {
            cr = 0.7;
        } else {
            cr = 0.3;
        }
        // Update stage 1 Q-values
        // Chosen action
        deltaC = R[i] - Q1[Y1[i] + 1];
        Q1[Y1[i] + 1] += eta_1 * (w * lam * cr * deltaC + lam * (1-w) * deltaC);

        // Unchosen action
        deltaU = R[i] - Q1[(1-Y1[i]) + 1];
        Q1[(1-Y1[i]) + 1] += eta_1 * w * lam * (1 - cr) * deltaU;
    }
    
    // Likelihood
    Y1 ~ bernoulli_logit( beta_1 * V1 );
    Y2 ~ bernoulli_logit( beta_2 * V2 );
    
    }

}