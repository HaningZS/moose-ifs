[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 40
  ny = 10
  xmin = 0.0
  xmax = 2.0
  ymin = 0.0
  ymax = 0.5
  elem_type = QUAD4
[]
[Variables]
  [c]
    initial_condition = 0.0
  []
[]
[Kernels]
  [time_derivative]
    type = TimeDerivative
    variable = c
  []
  [diffusion]
    type = Diffusion
    variable = c
    diffusivity = 5e-4
  []
  [decay]
    type = Reaction
    variable = c
    reaction_rate = 0.03
  []
[]
[BCs]
  [inlet]
    type = DirichletBC
    variable = c
    boundary = left
    value = 1.0
  []
[]
[Executioner]
  type = Transient
  solve_type = PJFNK
  start_time = 0.0
  end_time = 20.0
  dt = 0.1
  nl_abs_tol = 1e-8
[]
[Outputs]
  exodus = true
[]
