from statsmodels.stats.diagnostic import linear_reset
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def linear_regression(x, y, is_plot_residuals):
    """Build linear regression model

        input:
            x - x values
            y - y values
            is_plot - plot residuals

        output:
            y_pred - predicted y values
            k - slope of linear regression
            b - intercept of linear regression
            p_value - significance of linear regression model
            model_lin.rsquared - R^2 of linear regression model
            model_lin.aic - Akaike information criteria of linear regression model
    """

    #res = stats.linregress(x, y)
    # y_pred = res.intercept + res.slope * x

    ########### Building linear model ###########

    X = sm.add_constant(x)  # Is a function from statsmodels that adds a column of ones
                            # (a constant) to the data before regression. Add constant
                            # in a linear regression of the form: y=b0+b1*x
                            # column 1 is needed so that the model can estimate the intercept
                            # b0. If it is not added, the model will construct a regression without the intercept,
                            # meaning it will be forced through the origin (of coordinate system).

    model_lin = sm.OLS(y, X).fit()
    y_pred = model_lin.predict()
    residuals = y-y_pred    # Residuals of model

    # Coefficients
    b = model_lin.params[0]  # beta_0
    k = model_lin.params[1]  # beta_1


    # p-value for each coefficient
    p_value = model_lin.f_pvalue

    #p_intercept = p_values[0]
    #p_slope = p_values[1]

    reset_test(model_lin)

    if is_plot_residuals:
        plot_residuals_vs_log2_k(x,residuals, "linear")

    return y_pred, k, b, p_value, model_lin.rsquared, model_lin.aic

def reset_test(model_lin):
    """RESET test (by default quadratic and cubic terms)"""
    reset_test = linear_reset(model_lin, power=2, use_f=True)
    print("RESET-test linear model:", reset_test)
    p_value_reset = reset_test.pvalue
    if p_value_reset > 0.05:
        with open("RESET_RESULT.txt", "w", encoding="utf-8") as f:
            f.write("NOT OK")




def quadratic_regression(x, y, is_plot_residuals):
    """Build quadratic model

            input:
                x - x values
                y - y values
                is_plot - plot residuals

            output:
                y_quad_pred - predicted y values of quadratic_regression
                kefs - coefficients of quadratic regression b, b1*x1, b2*x2^2
                p_value - significance of quadratic regression model
                quadr_model_rsquared - R^2 of quadratic regression model
                quadr_model_aic - Akaike information criteria of quadratic regression model
        """

    #coeffs = np.polyfit(x, y, deg=2)
    #y_quadro_predicted = np.polyval(coeffs, x)
    #y_quadro_predicted = coeffs[0] *x * x + coeffs[1]*x + coeffs[2]

    ########### квадратичная модель
    X_quad = sm.add_constant(np.column_stack([x, x ** 2]))  # Is a function from statsmodels that adds a column of ones
                                                            # (a constant) to the data before regression. Add constant
                                                            # in a quadratic regression of the form: y=b0+b1*x+b2*x^2
                                                            # column 1 is needed so that the model can estimate the intercept
                                                            # b0. If it is not added, the model will construct a regression without the intercept,
                                                            # meaning it will be forced through the origin (of coordinate system).
    model_quad = sm.OLS(y, X_quad).fit()
    y_quad_pred = model_quad.predict(X_quad)
    residuals_quad = y - y_quad_pred                        # Residuals of model

    # Calculate R^2
    # from sklearn.metrics import r2_score
    # r2_squared = r2_score(y, y_quad_pred)
    quadr_model_rsquared = model_quad.rsquared

    quadr_model_aic = model_quad.aic
    p_value=model_quad.f_pvalue

    # коэффициенты
    kefs = [model_quad.params[0], model_quad.params[1], model_quad.params[2]]  # b, b1x1, b2x2^2
    p_x2 = model_quad.pvalues[2]
    print("p-value for coefficient near x^2:", p_x2)



    if is_plot_residuals:
        plot_residuals_vs_log2_k(x, residuals_quad, "quadratic")


    return y_quad_pred, kefs, p_value, quadr_model_rsquared, quadr_model_aic

def plot_residuals_vs_log2_k(x, residuals, title_mode):
    """Plot_residuals_vs_log2(k)
        input:
            x - x
            resuduals - resuduals
            title_mode - linear or quadratic
    """
    plt.scatter(x, residuals, color="blue")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("log2(k)")
    plt.ylabel("Residuals")
    plt.title("Residuals {0} vs log2(k)".format(title_mode))
    plt.show()

