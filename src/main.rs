use mpi::request::WaitGuard;
use mpi::traits::*;
use rand::prelude::SliceRandom;
use rand::Rng;

fn basic(comm: &mpi::topology::SystemCommunicator, rank: mpi::Rank, size: mpi::Rank) {
    let next_rank = (rank + 1) % size;
    let previous_rank = (rank - 1 + size) % size;

    let msg = vec![rank, 2 * rank, 4 * rank];
    mpi::request::scope(|scope| {
        let _sreq = WaitGuard::from(
            comm.process_at_rank(next_rank)
                .immediate_send(scope, &msg[..]),
        );

        let (msg, status) = comm.any_process().receive_vec();

        println!(
            "Process {} got message {:?}.\nStatus is: {:?}",
            rank, msg, status
        );
        let x = status.source_rank();
        assert_eq!(x, previous_rank);
        assert_eq!(vec![x, 2 * x, 4 * x], msg);

        let root_rank = 0;
        let root_process = comm.process_at_rank(root_rank);

        let mut a;
        if comm.rank() == root_rank {
            a = vec![2, 4, 8, 16];
            println!("Root broadcasting value: {:?}.", &a[..]);
        } else {
            a = vec![0; 4];
        }
        root_process.broadcast_into(&mut a[..]);
        println!("Rank {} received value: {:?}.", comm.rank(), &a[..]);
        assert_eq!(&a[..], &[2, 4, 8, 16]);
    });
}

fn tut2a(comm: &mpi::topology::SystemCommunicator, rank: mpi::Rank, size: mpi::Rank) {
    if rank == 0 {
        let data = vec![-1];
        print!("{rank}: Sending {:?}", data);
        comm.process_at_rank(1).send(&data);
    } else if rank == 1 {
        let (msg, status) = (*comm).any_process().receive_vec::<i32>();
        println!("{rank}: Recieved {:?}", msg);
    } else {
        println!("{rank}: I'm not included in this communication");
    }
}

fn tut2b(comm: &mpi::topology::SystemCommunicator, rank: mpi::Rank, size: mpi::Rank) {
    let mut ping_pong_count = 0;
    const MAX_SHOTS: i32 = 100;
    const NUM_PLAYERS: i32 = 2;
    let partner_rank = (rank + 1) % NUM_PLAYERS; //Where do I send

    while ping_pong_count < MAX_SHOTS {
        if (rank == ping_pong_count % NUM_PLAYERS) && rank < NUM_PLAYERS {
            ping_pong_count += 1;
            comm.process_at_rank(partner_rank).send(&ping_pong_count);
            println!("{rank}: shooting {ping_pong_count}");
        } else if rank < NUM_PLAYERS {
            let (msg, status) = (*comm).process_at_rank(partner_rank).receive::<i32>();
            println!("{rank}: receiving {msg}");
            ping_pong_count = msg;
        } else {
            println!("{rank}: I'm not playing");
        }
    }
}

fn tut2c(comm: &mpi::topology::SystemCommunicator, rank: mpi::Rank, size: mpi::Rank) {
    let mut token;
    if rank != 0 {
        (token, _) = (*comm).process_at_rank(rank - 1).receive::<i32>();
        println!("{rank}: received token: {token}");
    } else {
        token = -1;
        println!("{rank}: Initialize token(-1)");
    }

    comm.process_at_rank((rank + 1) % size).send(&token);
    println!(
        "{rank}: handing token: {token} over to {}",
        (rank + 1) % size
    );

    if rank == 0 {
        (token, _) = (*comm).process_at_rank(size - 1).receive::<i32>();
        println!("{rank}: received token: {token}");
    }
}

fn tut3a(comm: &mpi::topology::SystemCommunicator, rank: mpi::Rank, size: mpi::Rank) {
    const MAX_NUMBERS: u32 = 100;
    let mut rng = rand::thread_rng();

    if rank == 0 {
        let mut numbers: Vec<u32> = (0..MAX_NUMBERS).collect();
        numbers.shuffle(&mut rng);
        let numbers_sent = rng.gen_range(0..MAX_NUMBERS);
        comm.process_at_rank((rank + 1) % size)
            .send(&numbers[0..numbers_sent as usize]);
        println!(
            "{rank}: sending {} numbers to {}",
            numbers_sent,
            (rank + 1) % size
        );
    } else if rank == 1 {
        let (msg, status) = (*comm).process_at_rank(rank - 1).receive_vec::<i32>();
        println!(
            "{rank}: received  {} elements. STATUS: source_rank({}), tag: {}",
            msg.len(),
            status.source_rank(),
            status.tag()
        );
    } else {
        println!("{rank}: I'm not playing");
    }
}

fn tut5(comm: &mpi::topology::SystemCommunicator, rank: mpi::Rank, size: mpi::Rank) {
    let mut value;
    if rank == 0 {
        value = 100;
    } else {
        value = 0;
    }
    (comm).process_at_rank(0).broadcast_into(&mut value);

    println!("{rank}: My value is : {value}");
}

fn tut6(comm: &mpi::topology::SystemCommunicator, rank: mpi::Rank, size: mpi::Rank) {
    const MAX_NUMBERS: u32 = 20;

    let mut recvbuf: Vec<u32> = vec![0; MAX_NUMBERS as usize / size as usize];
    if rank == 0 {
        let numbers: Vec<u32> = (0..MAX_NUMBERS).collect();
        (comm)
            .process_at_rank(0)
            .scatter_into_root(&numbers, &mut recvbuf);
    } else {
        (comm).process_at_rank(0).scatter_into(&mut recvbuf);
    }

    println!("{rank}: My values are : {:?}", recvbuf);
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();

    //basic(&world,rank,size);
    //tut2a(&world, rank, size);
    //tut2b(&world, rank, size);
    //tut2c(&world, rank, size);
    //tut3a(&world, rank, size);
    //tut5(&world, rank, size);
    tut6(&world, rank, size);
}
