import scipy.stats as stats  # Import stats from scipy for statistical tests


class HypothesisTests:
    def two_t_test(self, group1, group2):
        # Perform a two-sample t-test between two groups
        # Assumes equal variance between the groups
        return stats.ttest_ind(a=group1, b=group2, equal_var=True)

    def anova_test(self, data_groups):
        # Perform one-way ANOVA test on multiple groups
        # Accepts a list of groups (arrays) and unpacks them as arguments to f_oneway function
        return stats.f_oneway(*data_groups)
