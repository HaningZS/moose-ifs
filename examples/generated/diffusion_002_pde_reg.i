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
  [diff]
    type = CoefDiffusion
    variable = c
    coef = 1.0
  []
  [time]
    type = TimeDerivative
    variable = c
  []
[]

[BCs]
  [top]
    type = DirichletBC
    variable = c
    boundary = top
    value = 2.0
  []
  [bottom]
    type = DirichletBC
    variable = c
    boundary = bottom
    value = 8.0
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
  start_time = 0.0
  end_time = 1.0
  dt = 0.1
  nl_abs_tol = 1e-8
[]

[Outputs]
  exodus = true
[]