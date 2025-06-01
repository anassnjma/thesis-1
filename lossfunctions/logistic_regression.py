import numpy as np


class LogisticRegression():
    @staticmethod
    def loss(theta, x, y, lambda_param=None):
        """Loss function for logistic regression with numerical stability"""
        exponent = - y * (x.dot(theta))
        # Clip exponent to prevent overflow: if exponent > 500, exp(exponent) -> inf
        exponent_clipped = np.clip(exponent, -500, 500)
        return np.sum(np.log(1 + np.exp(exponent_clipped))) / x.shape[0]

    @staticmethod
    def gradient(theta, x, y, lambda_param=None):
        """
        Gradient function for logistic regression with numerical stability.
        """
        exponent = y * (x.dot(theta))
        # Clip exponent to prevent overflow
        exponent_clipped = np.clip(exponent, -500, 500)
        
        denominator = 1 + np.exp(exponent_clipped)
        gradient_loss = - (np.transpose(x) @ (y / denominator)) / x.shape[0]

        # Reshape to handle case where x is csr_matrix
        gradient_loss = gradient_loss.reshape(theta.shape)

        return gradient_loss


class LogisticRegressionSinglePoint():
    @staticmethod
    def loss(theta, xi, yi, lambda_param=None):
        exponent = - yi * (xi.dot(theta))
        exponent_clipped = np.clip(exponent, -500, 500)
        return np.log(1 + np.exp(exponent_clipped))

    @staticmethod
    def gradient(theta, xi, yi, lambda_param=None):
        exponent = yi * (xi.dot(theta))
        exponent_clipped = np.clip(exponent, -500, 500)
        return - (yi * xi) / (1 + np.exp(exponent_clipped))


class LogisticRegressionRegular():
    @staticmethod
    def loss(theta, x, y, lambda_param):
        regularization = (lambda_param/2) * np.sum(theta*theta)
        return LogisticRegression.loss(theta, x, y) + regularization

    @staticmethod
    def gradient(theta, x, y, lambda_param):
        regularization = lambda_param * theta
        return LogisticRegression.gradient(theta, x, y) + regularization