data {
  // Metadata
  int<lower=1> T;  // Number of trials
  // int N;  // Number of participants 

  int<lower=0, upper=1> Y[T,2]; // choice data for each level (1 = right, 0 = left)
  int<lower=0, upper=1> O[T]; // outcome data for level 1 choice (1 = right, 0 = left)
  int<lower=0, upper=1> reward[T]; // trial reward
}
transformed data{
    // vectorize?
}
parameters {
  // Hyperparameters
  // (Add here when multiple-participant)

  // Subject-level parameters
  real eta1_pr; // first learning rate
  real eta2_pr; // second learning rate
  real beta1_pr; // first inv. temp
  real beta2_pr; // second inv. temp
  real w_pr; // weight
  real p_pr; // perseveration
}
transformed parameters {
  // Subject-level parameters
  real<lower=0, upper=1> eta1;
  real<lower=0, upper=1> eta2;
  real<lower=0, upper=20> beta1;
  real<lower=0, upper=20> beta2;
  real<lower=0, upper=1> w;
  real<lower=0, upper=1> p;

  beta1 = Phi_approx(beta1_pr) * 20;
  beta2 = Phi_approx(beta2_pr) * 20;
  eta1 = Phi_approx(eta1_pr);
  eta2 = Phi_approx(eta2_pr);
  p = Phi_approx(p_pr);
  w = Phi_approx(w_pr);
}
model {
    // define and initialize values
    vector[4] Q = rep_vector(0.5,4); // model-based Q values (Left: 1, 2, Right: 3, 4)
    vector[2] MF = rep_vector(0.5,2); // model-free values

    vector[T] deV1 = rep_vector(0, T); // difference in value, level 1
    vector[T] deV2 = rep_vector(0, T); // difference in value, level 2
    int m; // rep(a) value: 1 if a is a top-stage action and is the same one as was chosen on the previous trial

  // PRIORS
  // individual parameters
  eta1_pr ~ normal(0,1);
  eta2_pr ~ normal(0,1); 
  beta1_pr ~ normal(0,1); 
  beta2_pr ~ normal(0,1);
  p_pr ~ normal(0,1); 
  w_pr ~ normal(0,1);

  // loop through the trials
  for (i in 1:T) {

    // Compute hybrid value
    deV1[i] = w*0.4*(fmax(Q[4], Q[3]) - fmax(Q[2], Q[1])) + (1-w)*(MF[2] - MF[1]);

    // Choice likelihood for level 1
    if (i > 1) {
        // m = 1 (if first stage and same as previous trial), otherwise m = 0
        m = 1 ? (O[i-1] == O[i]): 0; // ternary conditional operator
        deV1[i] += p*m;
    }

    // update second hybrid values - choice likelihood for level 2
    deV2[i] = Q[2 + (O[i]*2)] - Q[1 + (O[i]*2)];

    // Y[i,2] -- say what was chosen in stage 2
    // O[i] -- tell us which stage that was in

    // Model-free update
    MF[Y[i,1]+1] += eta1*(reward[i] - Q[(O[i]*2) + (Y[i,2] + 1)]) + eta1*(Q[(O[i]*2) + (Y[i,2] + 1)] - MF[Y[i,1]+1]);

    // Update Q-values (MB)
    Q[(O[i]*2) + (Y[i,2] + 1)] += eta2 * (reward[i] - Q[(O[i]*2) + (Y[i,2] + 1)]);

  }
  // Assign likelihoods
  Y[:,1] ~ bernoulli_logit( beta1 * deV1);
  Y[:,2] ~ bernoulli_logit( beta2 * deV2 );
}
