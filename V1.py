import math
import pygame
import scipy

scale = 0.02
dt = 1
c = 1500 #speed of sound in m/s in water
g = pygame.Vector2(0, (9.81/scale))
P0 = 101325
class Particle:
    def __init__(self, diameter, mass, speed, acceleration, x_position, y_position, color):
        self.diameter = diameter
        self.mass = mass
        self.speed = pygame.Vector2(speed)
        self.acceleration = pygame.Vector2(acceleration)
        self.position = pygame.Vector2(x_position, y_position)
        self.radius = diameter / 2
        self.color = color
    def __lt__(self, other):
        if self.position.x != other.position.x:
            return self.position.x < other.position.x
        else:
            return self.position.y < other.position.y
    def movement(self, screen, dt):
        dampx = 0.1
        dampy = 0.1
        friction = 1
        self.speed += self.acceleration*dt

        self.position += self.speed*dt

        if self.position.x >= screen.get_width() - self.radius:
            self.position.x = screen.get_width() - self.radius
            self.speed.x *= -dampx
        elif self.position.x <= self.radius:
            self.position.x = self.radius
            self.speed.x *= -dampx
        if self.position.y >= screen.get_height() - self.radius:
            self.position.y = screen.get_height() - self.radius
            self.speed.y *= -dampy
            self.speed.x *= friction
        elif self.position.y <= self.radius:
            self.position.y = self.radius
            self.speed.y *= -dampy
    def neighbours(self,particles):
        N = []
        for p in particles:
            d = self.position.distance_to(p.position)
            r = self.radius
            if d <= r*5:
                N.append(p)
        return N
    def stat(self, particales):
        for other in particales:
            if self != other:
                p1 = particales.index(self)
                p2 = particales.index(other)
                distance = self.position.distance_to(other.position)
                print(f'the distance between particale {p1} and {p2} is {distance}')
    def P_stat(self, particles):
        print("|-----------------------------|")
        print(f'The radius : {self.radius}\n'
              f'The mass : {self.mass}\n'
              f'The x_speed is : {self.speed[0]}; The y_speed is : {self.speed[1]}; The speed is : {self.speed.magnitude()}\n'
              f'The xpos is :{self.position.x} , The ypos is:{self.position.y}\n'
              f'The particule id: {particles.index(self)}')
    def Color_change(self, particles, screen):
        H = screen.get_height()
        W = screen.get_width()

        center_x = W / 2
        center_y = H / 2

        for p in particles:
            dx = p.position.x - center_x
            dy = p.position.y - center_y
            distance_squared = dx**2 + dy**2

            if distance_squared <= 40**2:
                p.color = (255, 0, 0)
            elif 40**2 < distance_squared <= 80**2:
                p.color = (0, 255, 0)
            else:
                p.color = (0, 0, 255)
    def draw(self,screen):
        pygame.draw.circle(screen, self.color, (self.position.x,self.position.y), self.radius)
    def draw_soft_particle(self, screen):
        R = math.floor(self.radius)*50
        particle_surface = pygame.Surface((R * 2, R * 2), pygame.SRCALPHA)
        for r in range(R, 0, -1):
            alpha = int(100 * (r / R))
            pygame.draw.circle(particle_surface, (*self.color, alpha), (R,R), r)
        screen.blit(particle_surface, (self.position.x - R, self.position.y - R))




    """def heatmap(self, particles, screen):
        H = screen.get_height()
        W = screen.get_width()
        n = 5
        color = (255,255,255)
        particlesmatrix = [[0 for i in range(5)] for i in range(5)]
        for i in range(5):
            for j in range(5):
                for p in particles:
                    p.color = color
                    if particlesmatrix[i][j] <= 7:
                        color = (255,255,255)
                    elif 7 < particlesmatrix[i][j] <= 10:
                        color = (0,0,255)
                    elif 10 < particlesmatrix[i][j] <= 20:
                        color = (0,255,0)
                    else:
                        color = (255,0,0)
                    if i*(W/n) <= p.position.x <= (i+1)*(W/n) and j*(H/n) <= p.position.y <= (j+1)*(H/n):
                        particlesmatrix[i][j] += 1
        return particlesmatrix"""

def collision(particles):
    sorted_particles = sorted(particles)
    n = len(sorted_particles)
    activeparticles = []
    i = 0
    while i < n - 1:
            while i < n - 1 and abs(sorted_particles[i].position.x - sorted_particles[i+1].position.x) <= 20 and abs(sorted_particles[i].position.y - sorted_particles[i+1].position.y):
                    if activeparticles == [] or activeparticles[-1] != sorted_particles[i]:
                        activeparticles.append(sorted_particles[i])
                    activeparticles.append((sorted_particles[i+1]))
                    i += 1
            for self in activeparticles:
                for other in activeparticles:
                    if self != other:
                        distance = self.position.distance_to(other.position)
                        overlap = (self.radius + other.radius) - distance
                        if distance <= (other.radius + self.radius)/1:
                            correction = overlap/2
                            collision_normal = self.position - other.position
                            if collision_normal != pygame.Vector2(0,0):
                                collision_normal = collision_normal.normalize()
                            temp = other.speed.copy()
                            m1 = other.mass
                            m2 = self.mass
                            other.speed = 1*((m1 - m2)/(m1 + m2)*other.speed + 2*m2/(m1 + m2)*self.speed)
                            self.speed = 1*((2*m1)/(m1+m2)*temp + ((m2 - m1)/(m1 + m2))*self.speed)
                            other.position -= 1.2*correction*collision_normal
                            self.position += 1.2*correction*collision_normal
            activeparticles.clear()
            i += 1
    return n


def kernel(r,h):
    if h:
        x = r/h
        alpha = 1/(math.pi*h**3)
        if 0 <= x <= 1:
            return alpha*(1 - 3/2 * x**2 + 3 / 4 * x**3)
        elif 1 <= x <= 2:
            return alpha*1/4*(2 - x)**3
        else:
            return 0


def d_kernel(r,h):
    if h:
        x = r/h
        alpha = 1/(math.pi*h**4)
        if 0 <= x <= 1:
            return alpha*((9/4)*x**2 - 3*x)
        elif 1 <= x <= 2:
            return alpha*(-(3/4)*(2-x)**2)
        else:
            return 0

def detection_radius(screen, particles,h):
    for p in particles:
        pygame.draw.circle(screen , "green", (p.position.x, p.position.y), h, 2)

def Simulation_step(particles):
    for p in particles:
        p.speed += g
    for p in particles:
        xprev = p.position
        p.position += dt*p.speed



def doubledensityrelaxation(particles, h):
    k = 1
    k_near = 1
    density_0 = 1
    for i , p in enumerate(particles):
        density = 0
        density_near = 0
        for j , other in enumerate(particles):
            r_ij = p.position.distance_to(other.position)
            q = r_ij/h
            if q < 1:
                density += (1 - q)**2
                density_near += (1 - q)**3
        P = k*(density - density_0)
        P_near = k_near*density_near
        dx = pygame.Vector2(0,0)
        for j, other in enumerate(particles):
            r_ij = p.position.distance_to(other.position)
            if q < 1:
                rij_vector = p.position - other.position
                if r_ij:
                    rij_vector /= r_ij
                else:
                    rij_vector = pygame.Vector2(0,0)
                D = dt**2*(P(1-q) + P_near*(1-q)**2)*rij_vector
                other.position += D/2
                dx -= D/2
        p.position += dx

def Springadjastment(particles,L,h, l):
    gamma = 1
    alpha = 1
    for i , p in enumerate(particles):
        for j , other in enumerate(particles):
            r_ij = p.position.distance_to(other.position)
            q = r_ij/h
            if q < 1:
                if not L[i][j]:
                    L[i][j] = h
            d = gamma*L[i][j]
            if r_ij > l + d:
                L[i][j] = L[i][j] + dt*alpha*(r_ij - l - d)
            else:
                L[i][j] = L[i][j] - dt*alpha*(l - d - r_ij)
    for M in L:
        for m in M:
            if m > h:
                m = 0

def viscosity_impulse(particles, h):
    gamma = 1
    beta = 1
    for p in particles:
        for other in particles:
            r_ij = p.position.distance_to(other.position)
            q = r_ij/h
            r_ij_vect = (p.position - other.position)/r_ij
            if q < 1:
                u = (p.speed - other.speed).dot(r_ij_vect)
                if u > 0:
                    I = dt*(1 - q)*(gamma*u + beta*u**2)*r_ij_vect
                    p.speed -= I/2
                    other.speed += I/2


def Spring_Displacement(particles, L, h):
    n = len(L)
    k_spring = 1
    D = pygame.Vector2(0,0)
    for i , p in enumerate(particles):
        for j , other in enumerate(particles):
            distance = p.position.distance_to(other.position)
            r_ij_vector = (p.position - other.position)/(distance)

            D = dt**2*k_spring*(1-L[i][j]/h)*(L[i][j] - distance)*r_ij_vector
            p.position -= D/2
            other.position += D/2

def Particledensities(particles, h):
    D = []
    for p in particles:
        density_i = 0
        for other in particles:
            r_ij = p.position.distance_to(other.position)*scale
            density_i += other.mass*kernel(r_ij, h*scale)
        D.append(density_i) #densities in g/cm^3
    return D



"""def DerivativeParticleDensities(particles, h):
    D = []
    for i, p in enumerate(particles):
        derivative_density = 0
        vi = p.speed*scale
        for j, other in enumerate(particles):
            if i != j:
                vj = other.speed*scale
                r_ij = (p.position.distance_to(other.position))*scale
                r_vec_ij = (p.position - other.position)*scale
                if r_vec_ij:
                    r_unit_ij = r_vec_ij.normalize()
                    velocity_diff = vi - vj
                    kernel_gradient = d_kernel(r_ij, h*scale) * r_unit_ij
                    derivative_density += other.mass * (velocity_diff.dot(kernel_gradient))

        D.append(abs(derivative_density)) # in g/cm^3/s
    return D"""


def Particlepressures(particles, h):
    D = Particledensities(particles, h)
    density_0 = 1 #densitie of water in g/cm^3
    gamma = 7
    k = (density_0 * c**2) / gamma

    P = [
        max(k * ((rho / density_0) ** gamma - 1), 0)
        for rho in D
    ]

    return P


def modified_heat_map(particles, h):
    pressures = Particlepressures(particles, h)
    max_pressures = max(pressures)

    low_pressure_color = (41, 220, 214)
    medium_pressure_color = (0, 255, 0)
    high_pressure_color = (255, 255, 0)
    very_high_pressure_color = (255, 0, 0)
    for i , p in enumerate(particles):
        pressure = pressures[i]
        pressure_ration = pressure/max_pressures
        if pressure_ration >= 0.95:
            p.color = interpolate_color(high_pressure_color, very_high_pressure_color, (pressure_ration - 0.95) / 0.25)
        elif pressure_ration >= 0.93:
            p.color = interpolate_color(medium_pressure_color, high_pressure_color, (pressure_ration - 0.93) / 0.25)
        elif pressure_ration >= 0.92:
            p.color = interpolate_color(low_pressure_color, medium_pressure_color, (pressure_ration - 0.92) / 0.25)
        else:
            p.color = low_pressure_color




def normalize_forces(forces):
    max_force = max(f.length() for f in forces)
    if max_force > 0:
        normalized_forces = [(f / max_force) * 1000 for f in forces]
        return normalized_forces
    return forces

def Forcecalculator(particles, h):
    D = Particledensities(particles, h)
    P = Particlepressures(particles, h)
    Normalized_P = P  # Consider whether further normalization is needed here
    mu = 0.01  # Viscosity coefficient

    F = []
    n = len(particles)
    h_scaled = h * scale  # Cache the scaled smoothing length for performance

    for p in range(n):
        Fi = pygame.Vector2(0, 0)

        for other in range(n):
            if p != other:
                r_ij_vec = particles[p].position - particles[other].position
                r_ij = r_ij_vec.length() * scale

                if r_ij > 0:
                    r_ij_normalized = r_ij_vec.normalize()

                    if D[other] > 0 and D[p] > 0:
                        force_contribution = (particles[other].mass) * (
                                (Normalized_P[p] / (D[p]**2)) + (Normalized_P[other] / (D[other]**2))
                        ) * d_kernel(r_ij, h_scaled) * r_ij_normalized

                        Fi += force_contribution

                    vi = particles[p].speed * scale
                    vj = particles[other].speed * scale
                    F_ij_viscosity = mu * (vi - vj) * kernel(r_ij, h_scaled)
                    Fi += F_ij_viscosity

        F.append(-Fi)
    Normalized_Forces = [f/10**27 for f in F]
    return Normalized_Forces





def Pressure_Force(particles, h):
    F = Forcecalculator(particles, h)
    n = len(particles)
    for p in range(n):
        particles[p].acceleration += (F[p] / particles[p].mass) + g






def interpolate_color(color1, color2, ratio):
    return tuple([
        int(color1[i] + (color2[i] - color1[i]) * ratio)
        for i in range(3)
    ])

def density_heat_map(particles):
    max_neighbors = 0
    neighbors_count = []

    for p in particles:
        neighbors = p.neighbours(particles)
        count = len(neighbors)
        neighbors_count.append(count)
        if count > max_neighbors:
            max_neighbors = count

    low_density_color = (41, 220, 214)
    medium_density_color = (0, 255, 0)
    high_density_color = (255, 255, 0)
    very_high_density_color = (255, 0, 0)

    for i, p in enumerate(particles):
        count = neighbors_count[i]

        if max_neighbors > 0:
            density_ratio = count / max_neighbors

            if density_ratio >= 0.75:
                p.color = interpolate_color(high_density_color, very_high_density_color, (density_ratio - 0.75) / 0.25)
            elif density_ratio >= 0.5:
                p.color = interpolate_color(medium_density_color, high_density_color, (density_ratio - 0.5) / 0.25)
            elif density_ratio >= 0.25:
                p.color = interpolate_color(low_density_color, medium_density_color, (density_ratio - 0.25) / 0.25)
            else:
                p.color = low_density_color
        else:
            p.color = low_density_color








class Object:
    def __init__(self, length, width, moving, x_position, y_position):
        self.length = length
        self.width = width
        self.moving = moving
        self.position = pygame.Vector2(x_position,y_position)
    def objectcollision(self, particles):
        X = self.position.x
        Y = self.position.y
        W = self.width
        L = self.length

        for other in particles:
            r = other.radius
            x0 = other.position.x
            y0 = other.position.y
            x1 = x0 + other.speed.x
            y1 = y0 + other.speed.y

            if self.In_object(other.position):
                if x0 <= X:
                    other.speed.x = abs(other.speed.x)
                    print("Collision with left boundary (inside)")
                elif x0 >= X + L:
                    other.speed.x = -abs(other.speed.x)
                    print("Collision with right boundary (inside)")

                if y0 <= Y:
                    other.speed.y = abs(other.speed.y)
                    print("Collision with top boundary (inside)")
                elif y0 >= Y + W:
                    other.speed.y = -abs(other.speed.y)
                    print("Collision with bottom boundary (inside)")

            if not self.In_object(other.position):
                if (X - r <= x1 <= X + L) and (Y - r <= y1 <= Y + W):
                    if x1 >= X and x0 < X:
                        other.speed.x *= -1
                        print("Collision with left boundary")
                    elif x1 <= X + L + r and x0 > X + L - r:
                        other.speed.x *= -1
                        print("Collision with right boundary")

                    if y1 >= Y and y0 < Y:
                        other.speed.y *= -1
                        print("Collision with top boundary")
                    elif y1 <= Y + W and y0 > Y + W:
                        other.speed.y *= -1
                        print("Collision with bottom boundary")


    def In_object(self, position):
        x0 = self.position.x
        y0 = self.position.y
        W = self.width
        L = self.length
        x1 = position[0]
        y1 = position[1]

        return x0 <= x1 <= x0 + L and y0 <= y1 <= y0 + W



class rope:
    def __init__(self, length, position_x, position_y, speed, acceleration):
        self.length = length
        self.position = pygame.Vector2(position_x, position_y)
        self.speed = pygame.Vector2(speed)
        self.acceleration = pygame.Vector2(acceleration)
    def startcord(self):
        pass
    def O_stat(self):
        print("|******************************************************|")
        print(f'The cordinates are x: {self.position.x} ; y: {self.position.y} ; x+l: {self.position.x + self.length} ; y+w: {self.position.y + self.width}')


def apply_gaussian_filter_to_region(screen, sigma, rect):
    x, y, width, height = rect
    screen_array = pygame.surfarray.array3d(screen)
    roi = screen_array[x:x+width, y:y+height]
    filtered_roi = scipy.ndimage.gaussian_filter(roi, sigma=(sigma, sigma, 0))
    filtered_surface = pygame.surfarray.make_surface(filtered_roi)
    screen.blit(filtered_surface, (x, y))