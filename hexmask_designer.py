#!/usr/bin/env python

import pygame
import sys
import numpy as np
import xara
import os


# =============================================================================
def hex_grid_coords(nr=5, rr=1.0, rot=0.0):
    ij0 = np.linspace(-nr, nr, 2*nr+1)
    ii, jj = np.meshgrid(ij0, ij0)
    xx = rr * (ii + 0.5 * jj)
    yy = rr * jj * np.sqrt(3)/2
    cond = np.abs(ii + jj) <= nr
    return xx[cond], yy[cond]


# =============================================================================
def elt_grid_coords(rr=1.0):
    ''' -----------------------------------------------------------
    returns the (x,y) coordinates of active segments of the ELT

    Parameters:
    ----------
    - rr: the pitch of the segments

    Note:
    ----
    - for the actual ELT, the picth is equal to 1.4 meters
    - considers the segments falling under the spiders as missing
    ----------------------------------------------------------- '''
    nr = 18  # for the ELT, no choice
    no = 4   # idem for the central obstruction

    xx, yy = hex_grid_coords(nr=nr, rr=rr)
    xxo, yyo = hex_grid_coords(nr=no, rr=rr, rot=0.0)

    for ii, test in enumerate(xxo):
        throw = (xxo[ii] == xx) * (yyo[ii] == yy)
        xx = np.delete(xx, throw)
        yy = np.delete(yy, throw)

    keep = np.sqrt(xx**2 + yy**2) < (nr - 0.1) * rr * np.sqrt(3) / 2
    spider = np.logical_not(
        (yy == 0) + (yy == np.sqrt(3) * xx) + (yy == -np.sqrt(3) * xx))
    keep = keep * spider
    return xx[keep], yy[keep]


rr = 1.0   # radius of the rings
xxt, yyt = elt_grid_coords(rr)     # telescope grid coordinates
nseg = len(xxt)

# =============================================================================
#                         pygame specific setup
# =============================================================================
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
SALMON = (250, 128, 114)
AZURE = (0, 128, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

HOLE_COLOR = RED
TEL_COLOR = (0, 128, 128)

WSZ = 900  # window height size (actual is (2*WSZ x WSZ))
reso = 27  # resolution factor for the display

WIN_XSZ, WIN_YSZ = 2*WSZ, WSZ
p0 = np.array([WSZ//2, WSZ//2])
p1 = np.array([3*WSZ//2, WSZ//2])

FPS = 15           # refresh rate (15 frames/second)
mx, my = 0, 0      # mouse position
updt_flag = False  # flag signaling it's time to udate the uv display
clicked = False    # flag for when the moused is clicked
grabbed = -1       # index of the hole grabbed with the mouse

# the following 3 should adapt to WSZ!
hrad = 13          # hole radius on screen
uvrad = 10         # uv-splodge radius on screen
fsize = 15         # font size

pygame.init()
screen = pygame.display.set_mode((WIN_XSZ, WIN_YSZ), depth=32)
fps_clock = pygame.time.Clock()  # start the clock

# =============================================================================

# telescope draw coordinates
xxtd, yytd = (xxt * reso + p0[0]).astype(int), (yyt * reso + p0[1]).astype(int)

srects = []                 # Rectangles for telescope *segments*
hrects, holes = [], []     # Rectangles & Surfaces for *holes*
uvrects, uvsurfs = [], []  # Rectangles & Surfaces for uv-sample points

for ii in range(nseg):
    srects.append(pygame.draw.circle(
        screen, TEL_COLOR, (xxtd[ii], yytd[ii]), hrad, 0))

mask = []   # list of "real" mask hole coordinates
uvdc = []   # list of pixel uv-coordinates

mdl = None  # xara model
nh = 0      # number of mask "holes" when starting from scratch
nuv = 0     # number of baselines

mask_fname = "./mymask.txt"  # name of coordinate mask saved

if os.path.exists(mask_fname):
    print("found default mask configuration file!")
    mask = np.loadtxt(mask_fname)
    nh = len(mask)

    for ii in range(nh):
        holes.append(pygame.Surface((2*hrad, 2*hrad)))
        holes[ii].set_colorkey(BLACK)
        hrects.append(pygame.draw.circle(
            holes[ii], HOLE_COLOR, (hrad, hrad), hrad, 0))
        hrects[ii].center = p0 + mask[ii] * reso

    updt_flag = True

# breakpoint()
# ----------------------
# labels for the display
# ----------------------
lblfonts = pygame.font.Font('freesansbold.ttf', 2*fsize)
lblrects, lblsurfs = [], []

lblsurfs.append(lblfonts.render("Telescope aperture", True, BLACK, WHITE))
lblsurfs.append(lblfonts.render("Fourier-plane", True, BLACK, WHITE))

lblrects.append(lblsurfs[0].get_rect())
lblrects.append(lblsurfs[1].get_rect())

lblrects[0].center = p0[0], WSZ - 20
lblrects[1].center = p1[0], WSZ - 20


# =============================================================================
def draw_dot_collection(xx, yy, rad=10, color=(0, 0, 128)):
    ''' -----------------------------------------------------------------------
    Draws a set of identical disks at screen coordinates (xx, yy)

    Parameters:
    ----------
    - xx: array of horizontal pixel screen coordinates
    - yy: array of vertical pixel screen coordinates
    - rad: radius of the disk (default=  10 pixels)
    - color: RBG tuple
    ----------------------------------------------------------------------- '''
    global screen
    for ii in range(len(xx)):
        pygame.draw.circle(
            screen, color, (xx[ii], yy[ii]), rad, 0)


# =============================================================================
def draw_telescope_feature(xx, yy, reso=20, color=(0, 0, 128)):
    xxd = (p0[0] + xx * reso).astype(int)  # display pixel coordinates
    yyd = (p0[1] + yy * reso).astype(int)  # display pixel coordinates
    draw_dot_collection(xxd, yyd, rad=hrad, color=color)


# =============================================================================
def draw_uv_plane(uu, vv, reso=20, color=(128, 0, 0)):
    xxd = (p1[0] + uu * reso).astype(int)  # display pixel coordinates
    yyd = (p1[1] + vv * reso).astype(int)  # display pixel coordinates
    draw_dot_collection(xxd, yyd, rad=uvrad, color=color)


# =============================================================================
def snap_hole(xx, yy):
    ''' -----------------------------------------------------------------------
    Snap the location of holes: locate the most nearby valid aperture location

    Updates the mask coordinates holes and returns new coordinates to display

    Parameters:
    ----------
    - xx, yy: input coordinates

    Returns: index of telescope segment to snap to
    ----------------------------------------------------------------------- '''
    global srects, nseg
    for ii in range(nseg):
        if srects[ii].collidepoint(xx, yy):
            break
    return ii


draw_telescope_feature(xxt, yyt, reso=reso, color=TEL_COLOR)

pygame.display.update()

keepgoing = True


# =============================================================================
# =============================================================================
while keepgoing:
    # 1. capture events
    # -----------------
    for event in pygame.event.get():
        # -- right corner click --
        if event.type == pygame.QUIT:
            keepgoing = False

        # --------------------
        # -- keyboard event --
        # --------------------
        elif event.type == pygame.KEYDOWN:
            mmods = pygame.key.get_mods()  # capture modifiers
            if event.key == pygame.K_ESCAPE:
                keepgoing = False
                break

            elif event.key == pygame.K_s:  # save the mask
                if mdl is not None:
                    np.savetxt(mask_fname, mask, fmt="%+.6e %+.6e")

            elif event.key == pygame.K_a:  # add a sub-aperture
                if mmods & pygame.KMOD_LSHIFT:
                    print("test")
                else:
                    holes.append(pygame.Surface((2*hrad, 2*hrad)))
                    holes[nh].set_colorkey(BLACK)
                    hrects.append(pygame.draw.circle(
                        holes[nh], HOLE_COLOR, (hrad, hrad), hrad, 0))
                    snapped = snap_hole(mx, my)
                    hrects[nh].center = srects[snapped].center
                    if mask == []:
                        mask = np.array([[xxt[snapped], yyt[snapped]]])
                    else:
                        mask = np.append(
                            mask, [[xxt[snapped], yyt[snapped]]], axis=0)
                    nh += 1
                    updt_flag = True
                break

            elif event.key == pygame.K_d:  # delete a sub-aperture
                for ii in range(nh):
                    if hrects[ii].collidepoint(mx, my):
                        select = ii
                        break
                try:
                    holes.pop(ii)
                    hrects.pop(ii)
                    # mask.pop(ii)
                    mask = np.delete(mask, ii, axis=0)
                    updt_flag = True
                    nh -= 1
                except IndexError:
                    print("No hole left to remove!")
                pass

            elif event.key == pygame.K_SPACE:
                if nh > 3:
                    mdlf = xara.KPI(array=np.array(mask), ndgt=3)
                    mdlf.filter_baselines(mdlf.RED == 1)
                    print(mdlf.RED)

        # -----------------
        # -- mouse event --
        # -----------------
        elif event.type == pygame.MOUSEMOTION:
            mx, my = event.pos  # keep track of the mouse position
            if clicked:
                snapped = snap_hole(mx, my)
                hrects[grabbed].center = srects[snapped].center
                # mask[grabbed] = hrects[grabbed].center
                mask[grabbed] = [xxt[snapped], yyt[snapped]]

        elif event.type == pygame.MOUSEBUTTONUP:
            clicked = False
            updt_flag = True
            grabbed = -1

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mmods = pygame.key.get_mods()  # capture keyboard modifiers
            clicked = True
            grabbed = -1
            for ii in range(nh):
                if hrects[ii].collidepoint(mx, my):
                    grabbed = ii
                    break

    # 2. update the uv plane plot
    # ----------------------------
    # update the xara model
    if updt_flag:
        if nh > 1:
            mdl = xara.KPI(array=np.array(mask), ndgt=1)
            uvdc = (p1 + mdl.UVC * reso / 2).astype(int)
            # breakpoint()
            uvrects = []  # flush the list of rectangles
            uvsurfs = []

            # update the redundancy labels!
            uvfont = pygame.font.Font('freesansbold.ttf', fsize)
            for ii in range(mdl.nbuv):
                thiscolor = BLACK
                if mdl.RED[ii] == 1:
                    thiscolor = AZURE
                uvsurfs.append(
                    uvfont.render("%d" % (mdl.RED[ii]), True, thiscolor, WHITE))
                uvrects.append(uvsurfs[ii].get_rect())
                uvrects[ii].center = uvdc[ii]
                uvsurfs[ii].set_colorkey(WHITE)
        updt_flag = False

    # 3. refresh the display
    # ----------------------
    screen.fill(WHITE)
    draw_telescope_feature(xxt, yyt, reso=reso, color=TEL_COLOR)

    if mdl is not None:
        draw_uv_plane(np.array([0]), np.array([0]),
                      reso=reso, color=BLACK)
        draw_uv_plane(mdl.UVC[:, 0] / 2, mdl.UVC[:, 1] / 2,
                      reso=reso, color=SALMON)
        draw_uv_plane(-mdl.UVC[:, 0] / 2, -mdl.UVC[:, 1] / 2,
                      reso=reso, color=BLUE)

        # redundancy labels
        for ii in range(mdl.nbuv):
            screen.blit(uvsurfs[ii], uvrects[ii])

        # figure "titles"
        for ii, lbl in enumerate(lblrects):
            screen.blit(lblsurfs[ii], lblrects[ii])

    for ii in range(nh):
        screen.blit(holes[ii], hrects[ii])
    fps_clock.tick(FPS)
    pygame.display.update()

# this is the end
pygame.quit()
sys.exit()
