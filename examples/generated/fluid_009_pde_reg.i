[Mesh]
  [gen]
    type = GeneratedMeshGenerator
    dim = 2
    nx = 10
    ny = 10
  []
[]

[Variables]
  [c]
  []
[]

[Kernels]
  [time_deriv]
    type = TimeDerivative
    variable = c
  []
  [diffusion]
    type = CoefDiffusion
    variable = c
    coef = 0.0005
  []
  [reaction]
    type = CoefReaction
    variable = c
    coefficient = 0.03
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
    type = NeumannBC
    variable = c
    boundary = right
    value = 0.0
  []
  [top]
    type = NeumannBC
    variable = c
    boundary = top
    value = 0.0
  []
  [bottom]
    type = NeumannBC
    variable = c
    boundary = bottom
    value = 0.0
  []
[]

[ICs]
  [c_ic]
    type = ConstantIC
    variable = c
    value = 0.0
  []
[]

[Executioner]
  type = Transient
  num_steps = 10
  dt = 0.1
  solve_type = PJFNK
  nl_abs_tol = 1e-8
[]

[Outputs]
  exodus = true
[]