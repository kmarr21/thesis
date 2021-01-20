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
  p = Phi_approx(p);
  w = Phi_approx(w);
}
model {
  // define and initialize values
  vector[2] Qs1 = rep_vector(0.5,2); // stage 1 Q values
  vector[4] Qs2 = rep_vector(0.5,4); // stage 2 Q values

  vector[T] deV1 = rep_vector(0, T); // difference in value, level 1
  vector[T] deV2 = rep_vector(0, T); // difference in value, level 2

  int ix; // index variable
  real cr; // transition probability (common = 0.7, rare = 0.3)
  real deltaC; // for chosen action
  real deltaU; // for unchosen action
  
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

      // Compute dev1 for choice likelihoods
      deV1[i] = Qs1[1] - Qs1[2];

      // Determine stage 2 indices
      ix = 1 ? 1 : 3;

      // Compute dev2 for choice likelihoods
      deV2[i] = Qs2[ix] - Qs2[ix+1];

      // Update stage 2 Q-values
      Qs2[Y[i,2] + ix] += eta2 * (reward[i] - Qs2[Y[i,2] + ix]);

      // Determine transition probability
      if(Y[i,1] == O[i]) {
        cr = 0.7;
      }
      else {
        cr = 0.3;
      }
      // Update stage 1 Q-values
      // Chosen action
      deltaC = reward[i] - Qs1[Y[i,1] + 1];
      Qs1[Y[i,1] + 1] += eta1 * (w * cr * deltaC + (1-w) * deltaC);

      // Unchosen action
      deltaU = reward[i] - Qs1[(1-Y[i,1]) + 1];
      Qs1[(1-Y[i,1]) + 1] += eta1 * w * (1 - cr) * deltaU;
      
  }
  // Assign likelihoods
  Y[:,1] ~ bernoulli_logit( beta1 * deV1);
  Y[:,2] ~ bernoulli_logit( beta2 * deV2);
}
