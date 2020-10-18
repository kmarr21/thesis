data {
  // Metadata
  int<lower=1, upper=T> T;  // Number of trials
  // int N;  // Number of participants 

  int<lower=1, upper=2> Y[max(T),2]; // choice data for each level (2 = right, 1 = left)
  int<lower=0, upper=1> O[max(T)]; // outcome data for level 1 choice (1 = right, 0 = left)
  int<lower=0, upper=1> reward[max(T)]; // trial reward
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
  real lambda_pr; // 
  real beta1_pr; // first inv. temp
  real beta2_pr; // second inv. temp
  real w_pr; // weight
  real p_pr; // perseveration
}
transformed parameters {
  // Subject-level parameters
  real<lower=0, upper=1> eta1;
  real<lower=0, upper=1> eta2;
  real<lower=0, upper=1> lambda;
  real<lower=0, upper=20> beta1;
  real<lower=0, upper=20> beta2;
  real<lower=0, upper=1> w;
  real<lower=0, upper=1> p;

  beta1 = Phi_approx(beta1_pr) * 20;
  beta2 = Phi_approx(beta2_pr) * 20;
  eta1 = Phi_approx(eta1_pr);
  eta2 = Phi_approx(eta2_pr);
  p = Phi_approx(p);
  w = Phi_approx(w);
  lambda = Phi_approx(lambda);
}
model {
  // PRIORS
  // individual parameters
  eta1_pr ~ normal(0,1);
  eta2_pr ~ normal(0,1); 
  beta1_pr ~ normal(0,1); 
  beta2_pr ~ normal(0,1);
  p_pr ~ normal(0,1); 
  w_pr ~ normal(0,1);
  lambda_pr ~ normal(0,1);

  // define and initialize values
  vector[4] MB = rep_vector(0.5,4); // model-based values (Left: 1, 2, Right: 3, 4)
  vector[2] MF = rep_vector(0.5,2); // model-free values
  vector[2] hybrid = rep_vector(0.5,2); // hybrid values

  vector[T] deV1 = rep_vector(0, T); // difference in value, level 1
  vector[T] deV2 = rep_vector(0, T); // difference in value, level 2
  int m; // rep(a) value: 1 if a is a top-stage action and is the same one as was chosen on the previous trial
  
  // NOTES:
  // This now matches how I wrote my python hybrid model, but I'm wondering if I should change it to
  // more match what you wrote in your email: opinions on this?

  // loop through the trials
  for i in (1:T) {

    // Compute first hybrid value
    hybrid[1] = w*0.4*(fmax(MB[3], MB[4]) - fmax(MB[1], MB[2])) + (1-w)*(MF[2] - MF[1]);

    // Choice likelihood for level 1
    if (t == 1) {
      // first trial
      deV1[i] = hybrid[1];
    } else {
      // m = 1 (if first stage and same as previous trial), otherwise m = 0
      m = 1 ? (O[i-1] == O[i]): 0; // ternary conditional operator
      deV1[i] = hybrid[1] + (p * m);
    }
    deV1[i] = hybrid[1] + p*m;

    // update second hybrid values
    hybrid[2] = MB[(O[i]+ 4) + (2*O[i])] - MB[(O[i]+ 3) + (2*O[i])];

    // Choice likelihood for level 2
    deV2[i] = hybrid[2];

    // Update Q-values
    // Model-free update
    MF[Y[i,1]] += eta1*(MB[(O[i]+ 3) + (2*O[i])] - MF[Y[i,1]]) + eta1*lambda*(reward[i] - MB[(O[i]+ 3) + (2*O[i])]);
    // Model-based update
    MB[(O[i]+ 3) + (2*O[i])] += eta2 * (reward[i] - MB[(O[i]+ 3) + (2*O[i])]);

  }
  // Assign likelihoods
  Y[:,1] ~ bernoulli_logit( beta1 * deV1);
  Y[:,2] ~ bernoulli_logit( beta2 * deV2 );
}
