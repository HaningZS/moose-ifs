[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = 10
  xmin = 0
  xmax = 1
[]
[Variables]
  [u]
  []
[]
[Kernels]
  [time_derivative]
    type = TimeDerivative
    variable = u
  []
  [diffusion]
    type = Diffusion
    variable = u
  []
[]
[BCs]
  [top]
    type = DirichletBC
    variable = u
    boundary = right
    value = 2
  []
  [bottom]
    type = DirichletBC
    variable = u
    boundary = left
    value = 8
  []
[]
[Materials]
  [diff]
    type = GenericConstantMaterial
    prop_names = 'diffusivity'
    prop_values = 1.0
  []
[]
[Executioner]
  type = Transient
  num_steps = 10
  dt = 0.1
  solve_type = PJFNK
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'
  nl_abs_tol = 1e-8
[]
[Outputs]
  exodus = true
[]
