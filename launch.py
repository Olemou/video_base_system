def train(
    model,
    dataloader,
    sampler,
    optimizer,
    criterion,
    device,
    num_epochs,
    steps_per_epoch,
    start_epoch=0,
):

    logger.info("Initializing DataIterator...")
    data_iter = DataIterator(dataloader, sampler)

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        data_iter.set_epoch(epoch)

        epoch_loss = 0.0

        # =========================
        # DEFINE PHASES
        # =========================
        phases = [
            {
                "name": "head_only",
                "steps": int(0.3 * steps_per_epoch),
                "freeze": ["backbone", "transformer"],
                "unfreeze": ["head"],
            },
            {
                "name": "mid_unfreeze",
                "steps": int(0.3 * steps_per_epoch),
                "freeze": ["backbone"],
                "unfreeze": ["transformer", "head"],
            },
            {
                "name": "full_train",
                "steps": int(0.4 * steps_per_epoch),
                "freeze": [],
                "unfreeze": ["backbone", "transformer", "head"],
            },
        ]

        step_global = 0

        for phase in phases:
            logger.info(f"Starting phase: {phase['name']}")

            # -------------------------
            # FREEZE / UNFREEZE
            # -------------------------
            for name in phase["freeze"]:
                if hasattr(model, name):
                    set_trainable(getattr(model, name), False)

            for name in phase["unfreeze"]:
                if hasattr(model, name):
                    set_trainable(getattr(model, name), True)

            # -------------------------
            # PHASE TRAINING
            # -------------------------
            for step in range(phase["steps"]):
                try:
                    batch = data_iter.next(epoch)

                    if isinstance(batch, (list, tuple)):
                        inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    else:
                        inputs = batch.to(device)
                        targets = None

                    outputs = model(inputs)

                    if targets is not None:
                        loss = criterion(outputs, targets)
                    else:
                        loss = outputs

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    step_global += 1

                    if step_global % 50 == 0:
                        logger.info(
                            f"Epoch [{epoch+1}] Global Step [{step_global}/{steps_per_epoch}] "
                            f"Loss: {loss.item():.4f}"
                        )

                except Exception as e:
                    logger.error(f"Training step failed: {e}")
                    continue

        avg_loss = epoch_loss / steps_per_epoch
        logger.info(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")