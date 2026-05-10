[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 1
    nx = 10
  []
[]

[Variables]
  [c]
  []
[]

[Kernels]
  [time_derivative]
    type = TimeDerivative
    variable = c
  []
  [diffusion]
    type = CoefDiffusion
    variable = c
    coef = 0.5
  []
  [source]
    type = BodyForce
    variable = c
    value = 2.0
  []
[]

[BCs]
  [left]
    type = DirichletBC
    variable = c
    boundary = left
    value = 1.0
  []
  [right]
    type = DirichletBC
    variable = c
    boundary = right
    value = 1.0
  []
[]

[ICs]
  [c_ic]
    type = ConstantIC
    variable = c
    value = 1.0
  []
[]

[Executioner]
  type = Transient
  start_time = 0.0
  end_time = 1.0
  dt = 0.1
  nl_abs_tol = 1e-8
[]

[Outputs]
  exodus = true
[]