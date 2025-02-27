from __future__ import division
import collections

import sympy as sym

import inputs

# define symbolic variables
l, r = sym.var('l, r')

# should really use letters for variables so as not to confound with params!
mu, theta = sym.var('mu, theta')


class Model(object):
    """Class representing a matching model with two-sided heterogeneity."""

    def __init__(self, assortativity, workers, firms, production, params):
        """
        Create an instance of the Model class.

        Parameters
        ----------
        assortativity : str
            String defining the type of matching assortativity. Must be one of
            'positive' or 'negative'.
        workers : inputs.Input
            Instance of the inputs.Input class defining workers with
            heterogeneous skill levels.
        firms : inputs.Input
            Instance of the inputs.Input class defining firms with
            heterogeneous productivity.
        production : sympy.Basic
            Symbolic expression describing the production function.
        params : dict
            Dictionary of model parameters for the production function.

        """
        self.assortativity = assortativity
        self.workers = workers
        self.firms = firms
        self.F = production
        self.F_params = params

    @property
    def assortativity(self):
        """
        String defining the matching assortativty.

        :getter: Return the current matching assortativity
        :setter: Set a new matching assortativity.
        :type: str

        """
        return self._assortativity

    @assortativity.setter
    def assortativity(self, value):
        """Set new matching assortativity."""
        self._assortativity = self._validate_assortativity(value)

    @property
    def F(self):
        """
        Symbolic expression describing the available production technology.

        :getter: Return the current production function.
        :setter: Set a new production function.
        :type: sympy.Basic

        """
        return self._F

    @F.setter
    def F(self, value):
        """Set a new production function."""
        self._F = self._validate_production_function(value)

    @property
    def F_params(self):
        """
        Dictionary of parameters for the production function, F.

        :getter: Return the current parameter dictionary.
        :type: dict

        """
        return self._F_params

    @F_params.setter
    def F_params(self, value):
        """Set a new dictionary of parameters for F."""
        self._F_params = self._validate_F_params(value)

    @property
    def firms(self):
        """
        Instance of the inputs.Input class describing firms with heterogeneous
        productivity.

        :getter: Return current firms.
        :setter: Set new firms.
        :type: inputs.Input

        """
        return self._firms

    @firms.setter
    def firms(self, value):
        """Set new firms."""
        self._firms = self._validate_input(value)

    @property
    def Fx(self):
        """
        Symbolic expression for the marginal product of worker skill.

        :getter: Return the expression for the the marginal product of worker
        skill.
        :type: sympy.Basic

        """
        return sym.diff(self.F, self.workers.var)

    @property
    def Fxy(self):
        """
        Symbolic expression for the skill complementarity.

        :getter: Return the expression for the skill complementarity.
        :type: sympy.Basic

        """
        return sym.diff(self.F, self.workers.var, self.firms.var)

    @property
    def Flr(self):
        """
        Symbolic expression for the quantities complementarity.

        :getter: Return the expression for the quantities complementarity.
        :type: sympy.Basic

        """
        return sym.diff(self.F, l, r)

    @property
    def Fxr(self):
        """
        Symbolic expression for the managerial resource complementarity.

        :getter: Return the expression for managerial resource complementarity.
        :type: sympy.Basic

        """
        return sym.diff(self.F, self.workers.var, r)

    @property
    def Fyl(self):
        """
        Symbolic expression for the span-of-control complementarity.

        :getter: Return the expression for the span-of-control complementarity.
        :type: sympy.Basic

        """
        return sym.diff(self.F, self.firms.var, l)

    @property
    def matching(self):
        """
        Instance of the DifferentiableMatching class describing the matching
        equilibrium.

        :getter: Return the current DifferentiableMatching instance.
        :type: DifferentiableMatching

        """
        if self.assortativity == 'positive':
            return PositiveAssortativeMatching(self)
        else:
            return NegativeAssortativeMatching(self)

    @property
    def params(self):
        """
        Dictionary of model parameters.

        :getter: Return the current parameter dictionary.
        :type: dict

        """
        # model_params = dict(self.F_params.items() +
        #                     self.workers.params.items() +
        #                     self.firms.params.items())
        model_params = {**self.F_params, **self.workers.params, **self.firms.params}
        
        return self._order_params(model_params)

    @property
    def workers(self):
        """
        Instance of the inputs.Input class describing workers with
        heterogeneous skill.

        :getter: Return current workers.
        :setter: Set new workers.
        :type: inputs.Input

        """
        return self._workers

    @workers.setter
    def workers(self, value):
        """Set new workers."""
        self._workers = self._validate_input(value)

    @staticmethod
    def _order_params(params):
        """Cast a dictionary to an order dictionary."""
        return collections.OrderedDict(sorted(params.items()))

    @staticmethod
    def _validate_assortativity(value):
        """Validates the matching assortativity."""
        valid_assortativities = ['positive', 'negative']
        if not isinstance(value, str):
            mesg = "Attribute 'assortativity' must have type str, not {}."
            raise AttributeError(mesg.format(value.__class__))
        elif value not in valid_assortativities:
            mesg = "Attribute 'assortativity' must be in {}."
            raise AttributeError(mesg.format(valid_assortativities))
        else:
            return value

    @staticmethod
    def _validate_input(value):
        """Validates the worker and firm attributes."""
        if not isinstance(value, inputs.Input):
            mesg = ("Attributes 'workers' and 'firms' must have " +
                    "type inputs.Input, not {}.")
            raise AttributeError(mesg.format(value.__class__))
        else:
            return value

    @staticmethod
    def _validate_F_params(params):
        """Validates the dictionary of model parameters."""
        if not isinstance(params, dict):
            mesg = "Attribute 'params' must have type dict, not {}."
            raise AttributeError(mesg.format(params.__class__))
        else:
            return params

    def _validate_production_function(self, F):
        """Validates the production function attribute."""
        if not isinstance(F, sym.Basic):
            mesg = "Attribute 'F' must have type sympy.Basic, not {}."
            raise AttributeError(mesg.format(F.__class__))
        elif not {l, r} < F.atoms():
            mesg = "Attribute 'F' must be an expression of r and l."
            raise AttributeError(mesg)
        elif not {self.workers.var, self.firms.var} < F.atoms():
            mesg = ("Attribute 'F' must be an expression of workers.var and " +
                    "firm.var variables.")
            raise AttributeError(mesg)
        else:
            return F


class DifferentiableMatching(object):
    """Base class representing a differentiable matching system of ODEs."""

    def __init__(self, model):
        """
        Create an instance of the DifferentiableMatching class.

        Parameters
        model : model.Model
            Instance of the model.Model class representing a matching model
            with two-sided heterogeneity.

        """
        self.model = model

    @property
    def _subs(self):
        """
        Dictionary of variable substitutions

        :getter: Return the current dictionary of substitutions.
        :type: dict

        """
        return {self.model.firms.var: mu, l: theta, r: 1.0}

    @property
    def f(self):
        """
        Symbolic expression for intensive output.

        :getter: Return the current expression for intensive output.
        :type: sympy.Basic.

        """
        expr = (1 / r) * self.model.F
        return expr.subs(self._subs)

    @property
    def H(self):
        """
        Ratio of worker probability density to firm probability density.

        :getter: Return current density ratio.
        :type: sympy.Basic

        """
        return self.model.workers.pdf / self.model.firms.pdf

    @property
    def input_types(self):
        """
        Symbolic expression for complementarity between input types.

        :getter: Return the current expression for the complementarity.
        :type: sympy.Basic

        """
        return self.model.Fxy.subs(self._subs)

    @property
    def model(self):
        """
        Instance of the model.Model class representing a matching model
        with two-sided heterogeneity.

        :getter: Return the current model.Model instance.
        :setter: Set a new model.Model instance
        :type: model.Model

        """
        return self._model

    @model.setter
    def model(self, model):
        """Set a new model.Model instance."""
        self._model = self._validate_model(model)

    @property
    def mu_prime(self):
        """
        Differential equation describing the equilibrium matching between
        workers and firms.

        :getter: Return the current expression for mu prime.
        :type: sympy.Basic

        """
        raise NotImplementedError

    @property
    def profit(self):
        """
        Symbolic expression for profit earned by a firm.

        :getter: Return the current expression for profits.
        :type: sympy.Basic.

        """
        revenue = self.f
        costs = theta * self.wage
        return revenue - costs

    @property
    def quantities(self):
        """
        Symbolic expression for complementarity between input quantities.

        :getter: Return the current expression for the complementarity.
        :type: sympy.Basic

        """
        return self.model.Flr.subs(self._subs)

    @property
    def type_resource(self):
        """
        Symbolic expression for complementarity between worker type and
        firm resources.

        :getter: Return the current expression for the complementarity.
        :type: sympy.Basic

        """
        return self.model.Fxr.subs(self._subs)

    @property
    def span_of_control(self):
        """
        Symbolic expression for span-of-control complementarity.

        :getter: Return the current expression for the complementarity.
        :type: sympy.Basic

        """
        return self.model.Fyl.subs(self._subs)

    @property
    def theta_prime(self):
        """
        Differential equation describing the equilibrium firm size.

        :getter: Return the current expression for theta prime.
        :type: sympy.Basic

        """
        raise NotImplementedError

    @property
    def wage(self):
        """
        Symbolic expression for wages paid to workers.

        :getter: Return the current expression for wages.
        :type: sympy.Basic.

        """
        return sym.diff(self.f, theta)

    @staticmethod
    def _validate_model(model):
        """Validates the model attribute."""
        if not isinstance(model, Model):
            mesg = "Attribute 'model' must have type model.Model, not {}."
            raise AttributeError(mesg.format(model.__class__))
        else:
            return model


class NegativeAssortativeMatching(DifferentiableMatching):
    """Class representing a model with negative assortative matching."""

    @property
    def mu_prime(self):
        """
        Differential equation describing the equilibrium matching between
        workers and firms.

        :getter: Return the current expression for mu prime.
        :type: sympy.Basic

        """
        expr = -self.H / theta
        return expr.subs(self._subs)

    @property
    def theta_prime(self):
        """
        Differential equation describing the equilibrium firm size.

        :getter: Return the current expression for theta prime.
        :type: sympy.Basic

        """
        expr = -(self.H * self.model.Fyl + self.model.Fxr) / self.model.Flr
        return expr.subs(self._subs)


class PositiveAssortativeMatching(DifferentiableMatching):
    """Class representing a model with positive assortative matching."""

    @property
    def mu_prime(self):
        """
        Differential equation describing the equilibrium matching between
        workers and firms.

        :getter: Return the current expression for mu prime.
        :type: sympy.Basic

        """
        expr = self.H / theta
        return expr.subs(self._subs)

    @property
    def theta_prime(self):
        """
        Differential equation describing the equilibrium firm size.

        :getter: Return the current expression for theta prime.
        :type: sympy.Basic

        """
        expr = (self.H * self.model.Fyl - self.model.Fxr) / self.model.Flr
        return expr.subs(self._subs)
