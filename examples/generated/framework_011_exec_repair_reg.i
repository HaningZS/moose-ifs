[Mesh]
  type = GeneratedMesh
  dim = 1
  nx = 50
  xmin = 0
  xmax = 5
[]
[Variables]
  [u]
    initial_condition = 1.0
  []
[]
[Kernels]
  [diffusion]
    type = Diffusion
    variable = u
  []
  [source]
    type = BodyForce
    variable = u
    value = 2.0
  []
[]
[BCs]
  [left]
    type = DirichletBC
    variable = u
    boundary = left
    value = 1.0
  []
  [right]
    type = DirichletBC
    variable = u
    boundary = right
    value = 1.0
  []
[]
[Materials]
  [diff_coeff]
    type = GenericConstantMaterial
    prop_names = 'D'
    prop_values = 0.5
  []
[]
[Kernels]
  [diffusion]
    diffusivity = D
  []
[]
[Executioner]
  type = Transient
  num_steps = 20
  dt = 0.5
  solve_type = PJFNK
  nl_abs_tol = 1e-8
[]
[Outputs]
  exodus = true
[]
