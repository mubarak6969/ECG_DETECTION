(function () {
    const canvas = document.getElementById('bg-ecg');
    if (!canvas) return;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reducedMotion) return; // canvas is also hidden via CSS; skip the animation loop entirely

    const ctx = canvas.getContext('2d');
    let width, height, dpr;

    function resize() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener('resize', resize);

    // One stylized PQRST beat shape, repeated across the width.
    function beatShape(x0, baseline, scale) {
        return [
            [x0, baseline],
            [x0 + 8 * scale, baseline],
            [x0 + 12 * scale, baseline - 6 * scale],
            [x0 + 16 * scale, baseline + 3 * scale],
            [x0 + 20 * scale, baseline],
            [x0 + 26 * scale, baseline],
            [x0 + 30 * scale, baseline - 40 * scale],
            [x0 + 34 * scale, baseline + 18 * scale],
            [x0 + 38 * scale, baseline],
            [x0 + 46 * scale, baseline],
            [x0 + 52 * scale, baseline - 10 * scale],
            [x0 + 58 * scale, baseline],
            [x0 + 90 * scale, baseline],
        ];
    }

    let offset = 0;
    const speed = 0.6;

    function draw() {
        ctx.clearRect(0, 0, width, height);
        const rowCount = 3;
        for (let row = 0; row < rowCount; row++) {
            const baseline = (height / (rowCount + 1)) * (row + 1);
            const scale = 1.1;
            const beatWidth = 90 * scale;
            const startX = -((offset * (row + 1) * 0.4) % beatWidth) - beatWidth;

            ctx.beginPath();
            ctx.strokeStyle = `rgba(63, 208, 201, ${0.5 - row * 0.12})`;
            ctx.lineWidth = 1.4;

            let x = startX;
            let first = true;
            while (x < width + beatWidth) {
                const points = beatShape(x, baseline, scale);
                for (const [px, py] of points) {
                    if (first) {
                        ctx.moveTo(px, py);
                        first = false;
                    } else {
                        ctx.lineTo(px, py);
                    }
                }
                x += beatWidth;
            }
            ctx.stroke();
        }
        offset += speed;
        requestAnimationFrame(draw);
    }

    requestAnimationFrame(draw);
})();
